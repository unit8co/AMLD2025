import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', "")
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION', "")

# Database Configuration
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", ""), os.getenv("NEO4J_PASS", ""))
DATABASE_ = os.getenv("NEO4J_DB", "memgraph")

# Cache Configuration
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
CACHE_CLAIMS_DIR = os.getenv("CACHE_CLAIMS_DIR", "cache_claims")
CACHE_EMBEDDINGS_DIR = os.getenv("CACHE_EMBEDDINGS_DIR", "cache_embeddings")
CACHE_GRAPHS_DIR = os.getenv("CACHE_GRAPHS_DIR", "cache_graphs")

VECTOR_INDEX_NAME = "policy_vectors" 


# Entity and Relation Types
POSSIBLE_ENTITIES = [
    "BENEFIT",
    "CLAIM_PROCEDURE",
    "CLAUSE",
    "CONDITION",
    "COVERAGE",
    "DEDUCTIBLE",
    "DEFINITION",
    "ENDORSEMENT",
    "EVENT",
    "EXCESS",
    "EXCLUSION",
    "LIMIT",
    "ORGANIZATION",
    "PAYOUT",
    "PERIL",
    "PERSON",
    "POLICYHOLDER",
    "PREMIUM",
    "RISK_OBJECT",
    "SCHEDULE",
    "SECTION",
    "SERVICE",
    "SUBSECTION",
    "TERM",
]

POSSIBLE_RELATIONSHIPS = [
    "AMENDS",
    "APPLIES_TO",
    "CONTAINS",
    "COVERS",
    "DESCRIBES",
    "EXCLUDES",
    "FOLLOWS",
    "HAS_CERTIFICATE",
    "HAS_SCHEDULE",
    "IMPACTS",
    "LEADS_TO",
    "MENTIONS",
    "PAYABLE_FOR",
    "PRECEDES",
    "PROVIDES",
    "REFERENCES",
    "VALID_DURING",
] 