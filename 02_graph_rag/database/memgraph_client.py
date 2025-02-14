from typing import Dict, Any, List
from neo4j import GraphDatabase
from textwrap import dedent
import logging

from config.settings import POSSIBLE_RELATIONSHIPS, VECTOR_INDEX_NAME, POSSIBLE_ENTITIES
from config.retriever_settings import (
    VECTOR_SIMILARITY_THRESHOLD,
    MIN_IMPORTANCE_SCORE,
    MAX_RESULTS,
    MAX_HOP_DEPTH,
    IGNORE_RELATIONSHIP_TYPES,
    SIMILARITY_ADJUSTMENTS
)

class MemgraphClient:
    """A client for interacting with Memgraph database.
    
    This client provides comprehensive functionality for:
    - Managing graph database connections
    - Performing vector similarity search
    - Creating and querying graph structures
    - Managing nodes and relationships
    - Extracting subgraphs and context
    """
    def __init__(self, uri: str, auth: tuple, database: str = "memgraph"):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.database = database
        # Verify connectivity once
        self.driver.verify_connectivity()
        # Add cache dictionary for connected nodes
        self._connected_nodes_cache = {}

        # Create index in a separate transaction
        try:
            with self.driver.session() as session:
                session.run("CREATE INDEX ON :Node(type);")
        except Exception as e:
            logging.warning(f"Index creation warning (can be ignored if index exists): {e}")

    def close(self):
        """Close the driver connection."""
        self.driver.close()

    def clear_database(self):
        """Deletes all nodes and relationships."""
        self.driver.execute_query(
            "MATCH (n) DETACH DELETE n;",
            database_=self.database
        )

    def upsert_node(self, node: Dict) -> int:
        """Upsert a node with all properties and type label."""
        node_type = node.get("type", "Node")
        query = f"""
        MERGE (n:Node {{text: $text}})
        ON CREATE SET
            n:{node_type},
            n.type = $type,
            n.context = $context,
            n.description = $description,
            n.value = $value,
            n.applies_to = $applies_to,
            n.embedding = $embedding
        ON MATCH SET
            n.context = $context,
            n.description = $description,
            n.value = $value,
            n.applies_to = $applies_to,
            n.embedding = $embedding
        RETURN id(n)
        """
        result = self.driver.execute_query(query, **node)
        return result.records[0]["id(n)"]

    def upsert_relationship(self, subject_text: str, predicate: str, object_text: str) -> None:
        """Creates or updates a relationship between nodes."""
        query = dedent(f"""
            MERGE (source:Node {{ text: $subj_text }})
            ON CREATE SET source:Chunk
            MERGE (target:Node {{ text: $obj_text }})
            ON CREATE SET target:Chunk
            MERGE (source)-[r:`{predicate}`]->(target)
            RETURN r
        """)

        self.driver.execute_query(
            query,
            database_=self.database,
            subj_text=subject_text,
            obj_text=object_text
        )

    def find_similar_nodes(self, embedding: List[float], limit: int = MAX_RESULTS, min_similarity: float = VECTOR_SIMILARITY_THRESHOLD, min_importance: float = MIN_IMPORTANCE_SCORE) -> List[Dict[str, Any]]:
        """
        Hybrid search with metadata filtering and type-specific adjustments.
        
        Args:
            embedding: Vector embedding to search against
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            min_importance: Minimum importance score for nodes
            
        Returns:
            List of dictionaries containing node information and similarity scores
        """
        query = f"""
        CALL vector_search.search("{VECTOR_INDEX_NAME}", $limit, $embedding)
        YIELD node, similarity
        WITH node, similarity
        WHERE
            node.type IN $allowed_types
            AND similarity > $min_similarity
            AND coalesce(node.importance_score, 1.0) > $min_importance
        WITH 
            node,
            similarity,
            CASE node.type
                WHEN 'EXCLUSION' THEN $exclusion_adjustment * similarity
                WHEN 'LIMIT' THEN $limit_adjustment * similarity
                ELSE similarity
            END as adjusted_score
        RETURN 
            ID(node) as id,
            node.text as text,
            node.type as type,
            node.description as description,
            node.value as value,
            node.applies_to as applies_to,
            node.context as context,
            similarity,
            node.community as community,
            node.betweenness_centrality as centrality
        ORDER BY 
            adjusted_score DESC
        LIMIT $limit
        """
        
        try:
            result = self.driver.execute_query(
                query,
                embedding=embedding,
                limit=limit,
                allowed_types=POSSIBLE_ENTITIES,
                min_similarity=min_similarity,
                min_importance=min_importance,
                exclusion_adjustment=SIMILARITY_ADJUSTMENTS["EXCLUSION"],
                limit_adjustment=SIMILARITY_ADJUSTMENTS["LIMIT"]
            )
            
            # Log what we found
            logging.info(f"Found {len(result.records)} potential nodes")
            if result.records:
                logging.debug(f"First 3 results: {[dict(r) for r in result.records[:3]]}")
            
            return [dict(n) for n in result.records]
            
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return []

    def verify_vector_index(self, index_name: str = VECTOR_INDEX_NAME) -> bool:
        """Verify vector index exists and is properly configured"""
        query = "CALL vector_search.show_index_info() YIELD * RETURN *"
        result = self.driver.execute_query(query)
        
        for record in result.records:
            if record["index_name"] == index_name:
                logging.info(f"Found vector index: {record}")
                return True
        logging.warning("Vector index not found")
        return False

    def check_nodes_with_embeddings(self) -> None:
        """Check if there are any nodes with embeddings in the database."""
        query = """
        MATCH (n:Node)
        WHERE n.embedding IS NOT NULL
        RETURN count(n) as count
        """
        
        try:
            result = self.driver.execute_query(query, database_=self.database)
            count = result.records[0]["count"]
            logging.info(f"Found {count} nodes with embeddings")
            
            if count == 0:
                logging.warning("No nodes with embeddings found in the database")
                
            # Get a sample node if any exist
            if count > 0:
                sample_query = """
                MATCH (n:Node)
                WHERE n.embedding IS NOT NULL
                RETURN n
                LIMIT 1
                """
                sample = self.driver.execute_query(sample_query, database_=self.database)
                if sample.records:
                    node = sample.records[0]["n"]
                    logging.info(f"Sample node properties: {dict(node)}")
                    
        except Exception as e:
            logging.error(f"Failed to check nodes with embeddings: {e}")
            raise

    def upsert_relationship_by_ids(self, start_id, rel_type, end_id, properties=None):
        """
        Create a relationship between two nodes using their IDs.
        
        Args:
            start_id: ID of the start node
            rel_type: Type of relationship
            end_id: ID of the end node
            properties: Optional dictionary of relationship properties
        """
        try:
            if properties:
                props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
                query = f"""
                MATCH (start), (end)
                WHERE id(start) = $start_id AND id(end) = $end_id
                MERGE (start)-[r:{rel_type} {{{props_str}}}]->(end)
                RETURN id(r) as rel_id
                """
                params = {
                    "start_id": start_id,
                    "end_id": end_id,
                    **properties
                }
            else:
                query = """
                MATCH (start), (end)
                WHERE id(start) = $start_id AND id(end) = $end_id
                MERGE (start)-[r:%s]->(end)
                RETURN id(r) as rel_id
                """ % rel_type
                params = {
                    "start_id": start_id,
                    "end_id": end_id
                }
            
            result = self.driver.execute_query(query, params)
            if result.records:
                return result.records[0]["rel_id"]
            return None
            
        except Exception as e:
            logging.error(f"Error creating relationship: {e}")
            return None

    def update_node_properties(self, node_id: int, properties: Dict[str, Any]) -> None:
        """Update properties of an existing node"""
        query = f"""
        MATCH (n)
        WHERE id(n) = $node_id
        SET n += $properties
        """
        try:
            self.driver.execute_query(
                query,
                node_id=node_id,
                properties=properties
            )
        except Exception as e:
            logging.error(f"Failed to update node {node_id}: {e}") 

    def get_node_type(self, node_id: int) -> str:
        """Get the type of a node by its ID."""
        query = "MATCH (n) WHERE ID(n) = $node_id RETURN n.type"
        result = self.driver.execute_query(query, {"node_id": node_id})
        return result.records[0]["n.type"] if result.records else "Unknown"

    def extract_subgraph_for_nodes(self, nodes: List[Dict], max_depth: int = MAX_HOP_DEPTH, ignore_relationships: List[str] = IGNORE_RELATIONSHIP_TYPES) -> Dict:
        """
        Extract a subgraph containing the given nodes and their relationships.
        
        Args:
            nodes: List of node dictionaries with 'id' field
            max_depth: Maximum number of hops to traverse
            ignore_relationships: List of relationship types to ignore
            
        Returns:
            Dict containing the graph text representation
        """
        if not nodes:
            return {"graph_text": []}
        
        # Extract node IDs
        node_ids = [node["id"] for node in nodes]
        
        # Build relationship filter
        rel_filter = "" if not ignore_relationships else f"AND NOT ANY(rel IN r WHERE type(rel) IN {ignore_relationships})"
        
        query = f"""
            UNWIND $node_ids AS node_id
            MATCH (n) WHERE id(n) = node_id
            MATCH path = (n)-[r*1..{max_depth}]-(m)
            WHERE id(m) <> id(n)
            {rel_filter}

            WITH COLLECT(DISTINCT n) + COLLECT(DISTINCT m) AS all_nodes, 
                COLLECT(DISTINCT path) AS all_paths

            // Extract node descriptions while filtering empty values
            UNWIND all_nodes AS node
            WITH node, all_paths
            WHERE 
                COALESCE(node.description, "") <> "" OR 
                COALESCE(node.text, "") <> "" OR 
                COALESCE(TOSTRING(node.value), "N/A") <> "N/A"

            WITH 
                COLLECT(DISTINCT 
                    node.type + 
                    (CASE WHEN COALESCE(node.description, "") <> "" THEN ": '" + node.description + "'" ELSE "" END) + 
                    (CASE WHEN COALESCE(node.text, "") <> "" THEN " - '" + node.text + "'" ELSE "" END) + 
                    (CASE WHEN COALESCE(TOSTRING(node.value), "N/A") <> "N/A" THEN " - Value: " + TOSTRING(node.value) ELSE "" END)
                ) AS node_descriptions, 
                all_paths

            // Process paths
            UNWIND all_paths AS path
            UNWIND relationships(path) AS rel
            WITH node_descriptions, path, 
                startNode(rel) AS start_node, 
                endNode(rel) AS end_node, 
                type(rel) AS rel_type

            // Collect unique start node descriptions while filtering empty values
            WITH node_descriptions, end_node, rel_type, start_node
            WHERE 
                COALESCE(start_node.description, "") <> "" OR 
                COALESCE(start_node.text, "") <> "" OR 
                COALESCE(TOSTRING(start_node.value), "N/A") <> "N/A"

            WITH node_descriptions, end_node, rel_type, 
                COLLECT(DISTINCT 
                    (CASE WHEN COALESCE(start_node.description, "") <> "" THEN "'" + start_node.description + "'" ELSE "" END) + 
                    (CASE WHEN COALESCE(start_node.text, "") <> "" THEN " - '" + start_node.text + "'" ELSE "" END) + 
                    (CASE WHEN COALESCE(TOSTRING(start_node.value), "N/A") <> "N/A" THEN " - Value: " + TOSTRING(start_node.value) ELSE "" END)
                ) AS start_node_texts

            // Ensure start_node_texts is properly bound
            WITH node_descriptions, end_node, rel_type, start_node_texts
            WHERE SIZE(start_node_texts) > 0

            // Format OR-grouped nodes properly
            WITH node_descriptions, end_node, rel_type, start_node_texts,
                CASE 
                    WHEN SIZE(start_node_texts) > 1 THEN 
                        "(" + REDUCE(merged = "", text IN start_node_texts | 
                            CASE WHEN merged = "" THEN text ELSE merged + " OR " + text END) + ")"
                    ELSE 
                        HEAD(start_node_texts)
                END AS merged_start_nodes

            // Format final paths, filtering out empty descriptions
            WITH node_descriptions, 
                merged_start_nodes + " -[" + rel_type + "]-> " + 
                "(" + end_node.type + 
                (CASE WHEN COALESCE(end_node.description, "") <> "" THEN ": '" + end_node.description + "'" ELSE "" END) + 
                (CASE WHEN COALESCE(end_node.text, "") <> "" THEN " - '" + end_node.text + "'" ELSE "" END) + 
                (CASE WHEN COALESCE(TOSTRING(end_node.value), "N/A") <> "N/A" THEN " - Value: " + TOSTRING(end_node.value) ELSE "" END) + 
                ")" 
                AS grouped_path

            WITH node_descriptions, COLLECT(DISTINCT grouped_path) AS optimized_paths
            RETURN optimized_paths AS graph_text
        """
        
        try:
            result = self.driver.execute_query(query, {"node_ids": node_ids})
            if not result.records:
                return {"graph_text": []}
            
            # Deduplicate graph text
            graph_text = list(set(result.records[0]["graph_text"]))
            return {"graph_text": graph_text}
            
        except Exception as e:
            logging.error(f"Error extracting subgraph: {e}")
            return {"graph_text": []}

    def create_indexes(self):
        """Create insurance-specific relationship indexes"""
        for rel_type in POSSIBLE_RELATIONSHIPS:
            try:
                self.driver.execute_query(
                    f"CREATE INDEX ON :{rel_type}_RELATIONSHIP;"
                )
            except Exception as e:
                logging.debug(f"Index exists: {rel_type}")

    def create_vector_index(self, index_name: str, label: str, property_name: str, dimension: int):
        """Create a vector index using an explicit session"""
        query = f"""
        CREATE VECTOR INDEX {index_name} 
        ON :{label}({property_name}) 
        WITH CONFIG {{
            "dimension": {dimension}, 
            "metric": "cos",
            "capacity": 1000
        }}
        """
        try:
            # Create a new session for index creation
            with self.driver.session() as session:
                session.run(query)
            logging.info(f"Created vector index {index_name}")
        except Exception as e:
            logging.warning(f"Vector index creation warning: {e}")