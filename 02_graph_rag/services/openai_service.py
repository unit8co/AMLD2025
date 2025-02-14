import json
import logging
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import AzureOpenAI, OpenAI
import backoff

from config.settings import (
    POSSIBLE_ENTITIES, 
    POSSIBLE_RELATIONSHIPS, 
    CACHE_DIR,
    CACHE_CLAIMS_DIR,
    CACHE_EMBEDDINGS_DIR,
    CACHE_GRAPHS_DIR
)

from dotenv import load_dotenv
load_dotenv()

# Initialize cache directories from settings
CACHE_PATHS = {
    'default': Path(CACHE_DIR),
    'claims': Path(CACHE_CLAIMS_DIR),
    'embeddings': Path(CACHE_EMBEDDINGS_DIR),
    'graphs': Path(CACHE_GRAPHS_DIR)
}

# Create all cache directories
for path in CACHE_PATHS.values():
    path.mkdir(exist_ok=True)

# Constants
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))  # Dimension size for embeddings

def _get_cache_key(text: str, prefix: str = "") -> str:
    """Generate a cache key from input text."""
    return f"{prefix}_{hashlib.md5(text.encode()).hexdigest()}"

def _get_cached_result(cache_key: str, cache_type: str = 'default') -> Optional[Dict]:
    """Retrieve cached result if it exists.
    
    Args:
        cache_key: The key for the cached item
        cache_type: The type of cache to use ('default', 'claims', or 'embeddings')
    """
    cache_file = CACHE_PATHS[cache_type] / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to read cache file: {e}")
    return None

def _save_to_cache(cache_key: str, data: Any, cache_type: str = 'default') -> None:
    """Save result to cache.
    
    Args:
        cache_key: The key for the cached item
        data: The data to cache
        cache_type: The type of cache to use ('default', 'claims', or 'embeddings')
    """
    cache_file = CACHE_PATHS[cache_type] / f"{cache_key}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logging.warning(f"Failed to write to cache: {e}")

def get_llm_client():
    """Get the appropriate LLM client based on environment variables."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "deepseek":
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        )
    elif os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    else:
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

def get_embedding_client():
    """Get the client for embeddings (OpenAI only)."""
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    else:
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

def call_openai_with_retries(model: str, messages: List[Dict[str, str]]) -> str:
    """
    Call OpenAI API with configured model.
    
    Args:
        model: The model to use from environment configuration
        messages: List of message dictionaries
    
    Returns:
        The content of the response
    """
    try:
        client = get_llm_client()
        
        # Base parameters
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4000,
        }
        
        # Add response_format only for compatible models
        if model in ["gpt-4o", "deepseek-chat"]:
            kwargs["response_format"] = {"type": "json_object"}
        
        # Make the API call
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        
        # For models without json_object support, try to clean the response
        if "response_format" not in kwargs and not content.strip().startswith("{"):
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            # Validate JSON
            try:
                json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, wrap in a basic structure
                content = json.dumps({"nodes": [], "relationships": []})
        
        return content
            
    except Exception as e:
        logging.error(f"OpenAI API call failed: {str(e)}")
        # Return empty result structure on error
        return json.dumps({"nodes": [], "relationships": []})

def call_gpt_for_kg_extraction(text_chunk: str, context: str, model: str) -> dict:
    """Combined extraction of nodes and relationships"""
    cache_key = _get_cache_key(
        f"{text_chunk[:200]}_{context[:200]}_{model}_full", 
        prefix="kg_extraction"
    )
    
    # Check cache
    if cached := _get_cached_result(cache_key):
        return cached
    
    entity_list = "\n- ".join(POSSIBLE_ENTITIES)
    rel_list = "\n- ".join(POSSIBLE_RELATIONSHIPS)
    
    messages = [
        {
            "role": "system",
            "content": f"""Extract BOTH nodes and relationships in one JSON response. Use ONLY:
            Entities: {entity_list}
            Relationships: {rel_list}
            Format: {{"nodes": [...], "relationships": [...]}}"""
        },
        {
            "role": "user",
            "content": f"""Text: {text_chunk}
            Context: {context}
            Extract insurance policy knowledge graph elements as JSON with:
            - nodes: text, type, description, value, applies_to
            - relationships: subject_text, predicate, object_text"""
        }
    ]
    
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate and process
        if not isinstance(result.get("nodes", []), list):
            raise ValueError("Invalid nodes format")
        if not isinstance(result.get("relationships", []), list):
            raise ValueError("Invalid relationships format")
            
        # Process relationships
        for rel in result["relationships"]:
            rel_type = rel["predicate"].upper().replace(" ", "_")
            if rel_type not in POSSIBLE_RELATIONSHIPS:
                rel["predicate"] = "RELATED_TO"
        
        # Cache result
        _save_to_cache(cache_key, result)
        return result
        
    except Exception as e:
        logging.error(f"Combined extraction failed: {str(e)}")
        return {"nodes": [], "relationships": []}

@backoff.on_exception(backoff.expo, Exception, max_tries=3, logger=logging)
def get_embedding(text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    """Call OpenAI's Embeddings API with retries and fallbacks."""
    # Validate input
    if not text.strip():
        logging.warning("Empty text passed to get_embedding")
        return None

    # Check cache first
    cache_key = _get_cache_key(text, prefix="emb")
    cached_result = _get_cached_result(cache_key, cache_type='embeddings')
    if cached_result is not None:
        return cached_result

    try:
        client = get_embedding_client()
        emb_response = client.embeddings.create(
            input=text,
            model=model
        )
        embedding = emb_response.data[0].embedding
        _save_to_cache(cache_key, embedding, cache_type='embeddings')
        return embedding
    except Exception as e:
        logging.error(f"Embedding failed for model {model}: {str(e)}")
        # Fallback to smaller model
        if model != "text-embedding-3-small":
            return get_embedding(text, "text-embedding-3-small")
        return None

def get_model_name() -> str:
    """Get the appropriate model name based on the LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o")
    elif provider == "deepseek":
        return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 