import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
import openai
import numpy as np
import json
import hashlib
import re
from textwrap import dedent

# Load environment variables with defaults
load_dotenv()
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "neo4j")
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('policy-analysis')

class MemgraphClient:
    """Simplified Memgraph client with essential functions"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            MEMGRAPH_URI,
            auth=(MEMGRAPH_USER, MEMGRAPH_PASSWORD)
        )
        
    def find_similar_nodes(self, embedding: List[float], 
                          limit: int = 5,
                          min_similarity: float = 0.7,
                          min_importance: float = 0.5) -> List[Dict]:
        """Find similar nodes using vector search"""
        query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL AND n.importance >= $min_importance
        WITH n, cosineSimilarity(n.embedding, $embedding) AS similarity
        WHERE similarity >= $min_similarity
        RETURN n, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                "embedding": embedding,
                "limit": limit,
                "min_similarity": min_similarity,
                "min_importance": min_importance
            })
            
            return [{
                "id": record["n"].id,
                "labels": list(record["n"].labels),
                "properties": dict(record["n"]),
                "similarity": record["similarity"]
            } for record in result]

    def extract_subgraph_for_nodes(self, nodes: List[Dict], 
                                  max_depth: int = 3,
                                  ignore_relationships: List[str] = None) -> Dict:
        """Expand subgraph around similar nodes"""
        node_ids = [node["id"] for node in nodes]
        query = """
        MATCH path = (n)-[r*1..%d]-(m)
        WHERE id(n) IN $node_ids AND 
              type(r) NOT IN $ignore_rels AND
              id(m) IN $node_ids
        RETURN nodes(path) as nodes, relationships(path) as relationships
        """ % max_depth

        with self.driver.session() as session:
            result = session.run(query, {
                "node_ids": node_ids,
                "ignore_rels": ignore_relationships or []
            })
            
            nodes = set()
            relationships = []
            for record in result:
                nodes.update(record["nodes"])
                relationships.extend(record["relationships"])
            
            return {
                "nodes": [dict(node) for node in nodes],
                "relationships": [{
                    "type": rel.type,
                    "start": rel.start_node.id,
                    "end": rel.end_node.id,
                    "properties": dict(rel)
                } for rel in relationships],
                "graph_text": self._format_graph_text(nodes, relationships)
            }

    def _format_graph_text(self, nodes, relationships) -> List[str]:
        """Format graph elements for text context"""
        formatted = []
        for node in nodes:
            formatted.append(f"({node.labels[0]}) {node.get('text', 'Unnamed node')}")
        for rel in relationships:
            start = next(n for n in nodes if n.id == rel.start_node.id)
            end = next(n for n in nodes if n.id == rel.end_node.id)
            formatted.append(
                f"({start.labels[0]}) -[{rel.type}]-> ({end.labels[0]})"
            )
        return formatted

class PolicyRetriever:
    """Simplified version of the policy retriever for notebook use"""
    
    def __init__(self, memgraph_client: MemgraphClient):
        self.memgraph_client = memgraph_client
        self.llm_client = openai.Client(api_key=OPENAI_API_KEY)
        self.model = "gpt-4-turbo"
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def analyze_claim(self, claim_text: str) -> Dict:
        """Main analysis workflow"""
        # Get embedding
        embedding = self.get_embedding(claim_text)
        
        # Find similar nodes
        similar_nodes = self.memgraph_client.find_similar_nodes(
            embedding,
            limit=5,
            min_similarity=0.7
        )
        
        # Expand subgraph
        subgraph = self.memgraph_client.extract_subgraph_for_nodes(
            similar_nodes,
            max_depth=3
        )
        
        # Build context
        context = self._build_context(similar_nodes, subgraph)
        
        # Get LLM analysis
        return self._ask_llm(claim_text, context)
    
    def _build_context(self, nodes: List[Dict], subgraph: Dict) -> str:
        """Build policy context text"""
        node_text = "\n".join([
            f"({n['labels'][0]}) {n['properties'].get('text', '')}"
            for n in nodes
        ])
        rel_text = "\n".join(subgraph.get("graph_text", []))
        return f"Policy Context:\n{node_text}\n\nRelationships:\n{rel_text}"
    
    def _ask_llm(self, claim: str, context: str) -> Dict:
        """Get LLM analysis"""
        prompt = f"""
        Analyze this insurance claim against the provided policy context:
        
        Claim: {claim}
        
        Policy Context:
        {context}
        
        Return response in JSON format with:
        - coverage (boolean)
        - explanation (string)
        - limits (list)
        - exclusions (list)
        """
        
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

# Example usage
if __name__ == "__main__":
    # Initialize clients
    mg_client = MemgraphClient()
    retriever = PolicyRetriever(mg_client)
    
    # Example claim
    claim = "My car windshield was cracked by a rock on the highway"
    
    # Process claim
    analysis = retriever.analyze_claim(claim)
    print("\nAnalysis Result:")
    print(json.dumps(analysis, indent=2)) 