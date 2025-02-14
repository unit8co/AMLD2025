import os
import logging
import sys
from typing import List, Dict
import math
import numpy as np
import argparse

# Remove the sys.path modification code and use absolute imports
from config.settings import URI, AUTH, DATABASE_, POSSIBLE_RELATIONSHIPS, VECTOR_INDEX_NAME
from utils.text_processing import read_markdown, chunk_text, split_into_sentences, unique_everseen
from database.memgraph_client import MemgraphClient
from services.openai_service import call_gpt_for_kg_extraction, get_embedding, get_model_name

def setup_logging(verbose: bool = False):
    """Configure structured logging"""
    logging.basicConfig(level=logging.WARNING)  # Set root logger to warning
    
    # Progress logger (always shown)
    progress = logging.getLogger('progress')
    progress.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    progress.addHandler(handler)
    progress.propagate = False
    
    # Debug logger (only in verbose mode)
    if verbose:
        debug = logging.getLogger('debug')
        debug.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        debug.addHandler(handler)
        debug.propagate = False

def create_hierarchical_chunks(text: str) -> List[Dict[str, str]]:
    """Create larger chunks based on paragraph boundaries"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    target_size = 1500  # ~1000-1500 tokens
    min_size = 800  # Minimum chunk size
    
    for para in paragraphs:
        para_length = len(para)
        
        # Start new chunk if adding this paragraph would exceed target size
        if current_length + para_length > target_size and current_length >= min_size:
            chunks.append({
                "text": '\n\n'.join(current_chunk),
                "context": '\n\n'.join(current_chunk[-3:])  # Last 3 paragraphs as context
            })
            current_chunk = []
            current_length = 0
            
        current_chunk.append(para)
        current_length += para_length
    
    # Add remaining content
    if current_chunk:
        chunks.append({
            "text": '\n\n'.join(current_chunk),
            "context": '\n\n'.join(current_chunk[-3:])
        })
    
    return chunks

def cosine_similarity_simple(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

def add_similarity_relationships(mg_upserter, node_embeddings, node_id_map, progress):
    """Improved similarity analysis with type-specific thresholds"""
    progress.info("Adding similarity relationships between nodes...")
    type_thresholds = {
        "LIMIT": 0.92,
        "CONCEPT": 0.85,
        "CLAUSE": 0.88,
        "DEFINITION": 0.9
    }
    
    texts = list(node_embeddings.keys())
    
    # Use numpy for vectorized operations
    embeddings = np.array(list(node_embeddings.values()))
    
    # Calculate cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1)
    similarity_matrix = np.dot(embeddings, embeddings.T) / np.outer(norms, norms)
    
    # Process upper triangle of matrix
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = similarity_matrix[i,j]
            node_type = mg_upserter.get_node_type(node_id_map[texts[i]])
            
            if similarity >= type_thresholds.get(node_type, 0.85):
                # Add relationship with confidence score
                mg_upserter.upsert_relationship_by_ids(
                    node_id_map[texts[i]],
                    "SEMANTICALLY_SIMILAR",
                    node_id_map[texts[j]],
                    properties={
                        "similarity": float(similarity),
                        "comparison_type": f"{node_type}-{node_type}"
                    }
                )

def create_node_with_context(
    text: str,
    node_type: str, 
    context: str,
    node_embeddings: dict,
    mg_upserter,
    description: str = "",
    value: str = "",
    applies_to: str = ""
) -> tuple:
    """Helper function to create a node with context and return its ID and embedding."""
    # Create a rich text representation for embedding
    if node_type == "LIMIT":
        combined_text = f"{node_type}: {text}"
        if applies_to:
            combined_text += f" | Applies To: {applies_to}"
        if value:
            combined_text += f" | Value: {value}"
        if description:
            combined_text += f" | Description: {description}"
        if context:
            combined_text += f" | Context: {context}"
        combined_text = " | ".join([part for part in combined_text.split(" | ") if not part.endswith(": ")])
    else:
        combined_text = f"{node_type}: {text} | Description: {description} | Context: {context}"
    
    embedding = get_embedding(combined_text)
    
    node = {
        "text": text,
        "type": node_type,
        "context": context,
        "description": description,
        "value": value,
        "applies_to": applies_to if node_type == "LIMIT" else "",
        "embedding": embedding
    }
    
    node_id = mg_upserter.upsert_node(node)
    node_embeddings[text] = embedding
    return node_id, embedding

def get_related_nodes(mg_upserter, node_texts: List[str]) -> str:
    """Query with parameterization"""
    if not node_texts:
        return "No related nodes from previous chunks"
    
    # Parameterized query
    query = """
    MATCH (n)-[r]->(m)
    WHERE n.text IN $texts OR m.text IN $texts
    RETURN 
        n.text as subject, 
        type(r) as predicate, 
        m.text as object
    LIMIT 15
    """
    
    try:
        result = mg_upserter.driver.execute_query(
            query,
            texts=node_texts
        )
        return "\n".join(
            f"{rel['subject']} -[{rel['predicate']}]-> {rel['object']}"
            for rel in result.records
        )
    except Exception as e:
        logging.error(f"Related nodes query failed: {e}")
        return "Could not retrieve related nodes"

def perform_cross_chunk_analysis(mg_upserter, node_id_map, progress):
    """Analyze and connect related entities across chunks"""
    progress.info("Starting cross-chunk analysis...")
    
    try:
        # Diagnostic check for LIMIT nodes
        limit_check = mg_upserter.driver.execute_query("""
            MATCH (l:LIMIT)
            RETURN count(l) as limit_count, 
                    l.applies_to as applies_to
            LIMIT 5
        """)
        progress.info(f"LIMIT node check: {limit_check.records}")
        
        # 1. Connect related LIMIT nodes across chunks
        limit_connections = mg_upserter.driver.execute_query("""
            MATCH (l1:LIMIT), (l2:LIMIT)
            WHERE l1.applies_to = l2.applies_to AND id(l1) < id(l2)
            MERGE (l1)-[r:RELATED_LIMIT]->(l2)
            SET r.source = 'cross_chunk_analysis'
            RETURN count(r) as connections
        """)
        progress.info(f"Connected {limit_connections.records[0]['connections']} related limits across chunks")

        # 2. Connect concepts with similar embeddings across chunks
        concept_connections = mg_upserter.driver.execute_query(f"""
            MATCH (c:CONCEPT)
            WHERE c.embedding IS NOT NULL
            WITH c, c.embedding AS embedding
            CALL vector_search.search("{VECTOR_INDEX_NAME}", 5, embedding) 
            YIELD node AS similar_node, similarity
            WHERE similarity > 0.85 AND id(c) < id(similar_node)
            MERGE (c)-[r:SEMANTICALLY_SIMILAR]->(similar_node)
            SET r.source = 'cross_chunk_analysis'
            RETURN count(r) as connections
        """)
        progress.info(f"Connected {concept_connections.records[0]['connections']} similar concepts across chunks")

        # 3. Connect clauses to their sub-clauses across chunks
        clause_hierarchy = mg_upserter.driver.execute_query("""
            MATCH (parent:CLAUSE), (child:CLAUSE)
            WHERE child.text CONTAINS parent.text 
                AND child.hierarchy_level = parent.hierarchy_level + 1
            MERGE (parent)-[r:HAS_SUBCLAUSE]->(child)
            SET r.source = 'cross_chunk_analysis'
            RETURN count(r) as connections
        """)
        progress.info(f"Connected {clause_hierarchy.records[0]['connections']} clause hierarchies across chunks")

        # 4. Create global relationships between core concepts
        core_concepts = mg_upserter.driver.execute_query("""
            MATCH (c:CONCEPT)
            WHERE c.name IN ['insurance', 'coverage', 'premium', 'claim']
            WITH collect(c) AS concepts
            UNWIND concepts AS c1
            UNWIND concepts AS c2
            WHERE id(c1) < id(c2)
            MERGE (c1)-[r:RELATED_CONCEPT]->(c2)
            SET r.global_relationship = true
            RETURN count(r) as connections
        """)
        progress.info(f"Created {core_concepts.records[0]['connections']} global concept relationships")

    except Exception as e:
        progress.error(f"Cross-chunk analysis failed: {e}")

def enrich_global_properties(mg_upserter, progress):
    """Add global properties and indexes"""
    progress.info("Adding global graph properties...")
    try:
        # Add centrality scores
        mg_upserter.driver.execute_query("""
            CALL betweenness_centrality.set()
            YIELD node, betweenness_centrality
            SET node.betweenness_centrality = betweenness_centrality
        """)
        
        # Add community detection
        mg_upserter.driver.execute_query("""
            CALL community_detection()
            YIELD node, community
            SET node.community = community
        """)
        
        progress.info("Added global graph properties")
    except Exception as e:
        progress.error(f"Global property enrichment failed: {e}")

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Build insurance policy knowledge graph")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logging")
    args = parser.parse_args()
    
    # Configure logging
    setup_logging(verbose=args.verbose)
    progress = logging.getLogger('progress')
    debug = logging.getLogger('debug')
    
    try:
        progress.info("Starting knowledge graph creation...")
        debug.debug("Database connection details: %s", URI)
        
        # 1) Read Markdown file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        md_path = os.path.join(project_root, "data", "llamaparse", "motor_2021.md")
        text = read_markdown(md_path)
        progress.info(f"Read text file, length: {len(text)}")

        # 2) Create hierarchical chunks with context
        chunks_with_context = create_hierarchical_chunks(text)
        progress.info("Created %d chunks with context", len(chunks_with_context))
        if not chunks_with_context:
            progress.error("No chunks created! Exiting...")
            return

        # 3) Initialize Memgraph Upserter with debug
        try:
            mg_upserter = MemgraphClient(uri=URI, auth=AUTH, database=DATABASE_)
            progress.info("Successfully connected to database")
        except Exception as e:
            progress.error(f"Failed to connect to database: {e}")
            return

        # Create vector index before processing data
        if not mg_upserter.verify_vector_index(VECTOR_INDEX_NAME):
            mg_upserter.create_vector_index(
                index_name=VECTOR_INDEX_NAME,
                label="Node",
                property_name="embedding",
                dimension=1536  # Match OpenAI embedding dimension
            )

        # Clear existing data if CLEAR_DATABASE env var is set to "true"
        if os.getenv("CLEAR_DATABASE", "false").lower() == "true":
            progress.info("Clearing existing database...")
            mg_upserter.clear_database()

        # Keep track of all node IDs across chunks
        node_id_map = {}
        node_embeddings = {}

        # 4) Process each chunk with detailed logging
        for i, chunk_data in enumerate(chunks_with_context):
            progress.info(f"\nProcessing chunk {i+1}/{len(chunks_with_context)}...")
            progress.info(f"Chunk text length: {len(chunk_data['text'])}")
            
            debug.debug("Full chunk text: %s", chunk_data['text'])
            debug.debug("Full chunk context: %s", chunk_data['context'])
            
            try:
                # Single extraction call
                extraction = call_gpt_for_kg_extraction(
                    text_chunk=chunk_data["text"],
                    context=chunk_data["context"],
                    model=get_model_name()
                )
                
                # Process nodes and relationships from single response
                nodes = extraction.get("nodes", [])
                relationships = extraction.get("relationships", [])
                
                # Process nodes
                for node in nodes:
                    node_id, _ = create_node_with_context(
                        text=node.get("text", ""),
                        node_type=node.get("type", "CONCEPT"),
                        context=chunk_data["context"],
                        node_embeddings=node_embeddings,
                        mg_upserter=mg_upserter,
                        description=node.get("description", ""),
                        value=node.get("value", ""),
                        applies_to=node.get("applies_to", "")
                    )
                    node_id_map[node.get("text")] = node_id  # Store node ID by text

                # Process relationships
                for rel in relationships:
                    subject_id = node_id_map.get(rel["subject_text"])
                    object_id = node_id_map.get(rel["object_text"])
                    
                    if subject_id and object_id:
                        mg_upserter.upsert_relationship_by_ids(
                            subject_id,
                            rel["predicate"],
                            object_id,
                            properties={"source": "chunk_processing"}
                        )

            except Exception as e:
                progress.error(f"Chunk processing failed: {str(e)}")
                debug.exception("Full error details:")
                continue

            # Log progress for this chunk
            progress.info(f"Processed chunk {i+1}, current total nodes: {len(node_id_map)}")

        # Add similarity relationships after processing all chunks
        add_similarity_relationships(mg_upserter, node_embeddings, node_id_map, progress)

        # Cross-chunk analysis
        perform_cross_chunk_analysis(mg_upserter, node_id_map, progress)
        enrich_global_properties(mg_upserter, progress)

        # Create indexes after data insertion for better performance
        mg_upserter.create_indexes()

        # Verification query with explicit relationship patterns
        verify_query = """
        MATCH (n)-[r]->(m)
        WITH count(r) as rel_count
        MATCH (n)
        RETURN rel_count, count(DISTINCT n) as node_count
        """
        result = mg_upserter.driver.execute_query(verify_query)
        stats = result.records[0] if result.records else {"rel_count": 0, "node_count": 0}
        progress.info(f"Final graph stats - Nodes: {stats['node_count']}, Relationships: {stats['rel_count']}")

        # Close the driver
        mg_upserter.close()
        progress.info("All done!")

    except Exception as e:
        progress.error("Fatal error in main process")
        debug.exception("Full error traceback:")
        raise

if __name__ == "__main__":
    main() 