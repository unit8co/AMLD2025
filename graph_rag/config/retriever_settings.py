"""
Configuration settings for the retriever component.
This file contains all parameters that influence the retrieval process.
"""

# Vector Search Settings
VECTOR_SIMILARITY_THRESHOLD = 0.40  # Minimum similarity score for vector search results
MIN_IMPORTANCE_SCORE = 0.0  # Minimum importance score for nodes to be considered
MAX_RESULTS = 7  # Maximum number of similar nodes to retrieve

# Graph Traversal Settings
MAX_HOP_DEPTH = 5  # Maximum number of hops when extracting subgraph relationships
IGNORE_RELATIONSHIP_TYPES = ["SEMANTICALLY_SIMILAR"]  # Relationship types to ignore during traversal

# Type-Specific Similarity Adjustments
SIMILARITY_ADJUSTMENTS = {
    "EXCLUSION": 0.8,  # Reduce importance of exclusions by 20%
    "LIMIT": 1.2,  # Boost importance of limits by 20%
}

# LLM Analysis Settings
SYSTEM_PROMPT_TEMPERATURE = 0.0  # Temperature for initial claim analysis
MAX_RETRIES = 3  # Maximum number of retries for LLM calls
RETRY_TEMPERATURE_INCREMENT = 0.1  # Temperature increment on each retry

# Cache Settings
ENABLE_CLAIM_GRAPH_CACHE = True  # Whether to cache extracted claim graphs
ENABLE_EMBEDDING_CACHE = True  # Whether to cache claim embeddings

# Logging Settings
LOG_SIMILAR_NODES = True  # Whether to log found similar nodes
LOG_SUBGRAPH_EXTRACTION = True  # Whether to log subgraph extraction details
DEBUG_FIRST_N_RESULTS = 3  # Number of top results to log in debug mode 