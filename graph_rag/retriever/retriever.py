from typing import List, Dict, Any
from textwrap import dedent
import logging
import json
import hashlib
import re

from database.memgraph_client import MemgraphClient
from services.openai_service import get_embedding, get_llm_client, get_model_name, _get_cache_key, _get_cached_result, _save_to_cache, call_openai_with_retries
from config.settings import POSSIBLE_ENTITIES, POSSIBLE_RELATIONSHIPS
from config.retriever_settings import (
    VECTOR_SIMILARITY_THRESHOLD,
    MIN_IMPORTANCE_SCORE,
    MAX_RESULTS,
    MAX_HOP_DEPTH,
    IGNORE_RELATIONSHIP_TYPES,
    SYSTEM_PROMPT_TEMPERATURE,
    MAX_RETRIES,
    RETRY_TEMPERATURE_INCREMENT,
    ENABLE_CLAIM_GRAPH_CACHE,
    ENABLE_EMBEDDING_CACHE,
    LOG_SIMILAR_NODES,
    LOG_SUBGRAPH_EXTRACTION,
    DEBUG_FIRST_N_RESULTS
)

# Create loggers
progress_logger = logging.getLogger('progress')
debug_logger = logging.getLogger('debug')

class PolicyRetriever:
    """Retrieves relevant policy information using vector similarity search."""
    
    def __init__(self, memgraph_client: MemgraphClient, verbose: bool = False):
        self.memgraph_client = memgraph_client
        self.verbose = verbose
        self.system_prompt = """**Insurance Policy Analysis Protocol**
        
        1. **Claim-Policy Mapping**
        - Match each claim element to EXACT policy clauses
        - Require verbatim text matches for conditions
        
        2. **Exclusion Check**
        - Check ALL exclusions in context
        - Apply exclusion if ANY match
        
        3. **Limit Application**
        - Apply MOST SPECIFIC limit first
        - Sum applicable limits
        
        4. **Coverage Requirements**
        - Verify ALL required conditions met
        - Reject if ANY missing
        
        5. **Decision Framework**
        IF ANY exclusion applies → Deny
        ELIF any condition unmet → Deny
        ELIF claim > limit → Partial coverage
        ELSE → Full coverage
        
        **Output Requirements**
        - Cite EXACT policy text for decisions
        - List MISSING requirements if denied
        - Calculate SPECIFIC amounts if limited"""
        self.llm_client = get_llm_client()
        self.model = get_model_name()

    def _log(self, message: str, level: str = "info", is_progress: bool = False):
        """Centralized logging function that respects verbose mode."""
        if is_progress:
            # Progress messages go only to progress logger
            if level == "error":
                progress_logger.error(message)
            elif level == "warning":
                progress_logger.warning(message)
            else:
                progress_logger.info(message)
        elif self.verbose:
            # Debug messages go only to debug logger when in verbose mode
            if level == "error":
                debug_logger.error(message)
            elif level == "warning":
                debug_logger.warning(message)
            else:
                debug_logger.info(message)

    def _build_prompt(self, claim_text: str, context: str) -> str:
        """Build the prompt with explicit format requirements"""
        format_example = json.dumps({
            "coverage": True,
            "explanation": "Detailed explanation...",
            "limits": [{
                "type": "Limit Type",
                "amount": 5000,
                "unit": "USD",
                "applies_to": "Damage Type",
                "source": "CLAUSE 2.1"
            }],
            "exclusions": [{
                "text": "Exclusion text",
                "applies": True,
                "reason": "Why it applies",
                "source": "EXCLUSION 3"
            }],
            "description": "Brief claim summary"
        }, indent=2)

        prompt_v2 = f"""You are an insurance policy analyzer. Your task is to analyze insurance claims 
        against policy documentation and determine coverage. For each claim, pay attention to the user question. 
        Only answer that question and return the coverage value of that question. 

        # CLAIM:
        {claim_text}

        # POLICY CONTEXT:
        {context}

        Analyze the claim and provide a response in EXACTLY this JSON format:
        {format_example}
        
        # Notes
        - The risk event is very important. For example, users are only entitled to overnight accomodation and alternative transport if they suffered an accident. Other risk events do not unlock that benefit.
        - Pay attention to the user question. For example, if the user is requesting compensation for glass damage, the user can only get full compensation if using an approved repairer. Otherwise, the amount is capped at £175. 
        - Pay attention to all details in the claim. For example, if the user claim is about fitted equipment, then the user can only get full compensation if it was fitted by the manufacturer. Otherwise, the amount is capped at £500.
        """

        prompt_v1 = f"""You are an insurance claim analyst. Your task is to analyze a claim against the relevant policy information.

        # CLAIM:
        {claim_text}

        # POLICY CONTEXT:
        {context}

        Analyze the claim and provide a response in EXACTLY this JSON format:
        {format_example}

        Requirements:
        1. "limits" and "exclusions" must be arrays (use empty arrays if none apply)
        2. Always include "source" fields referencing policy clauses
        3. For exclusions, specify both "applies" boolean and "reason"
        4. Amounts must be numbers (null if not applicable)"""
        
        prompt = prompt_v2

        self._log("Generated prompt:", level="info", is_progress=False)
        self._log("-" * 80, level="info", is_progress=False)
        self._log(prompt, level="info", is_progress=False)
        self._log("-" * 80, level="info", is_progress=False)

        return prompt

    def build_context_text(self, similar_nodes: List[Dict[str, Any]], connected_data: Dict, claim_graph: str = None) -> str:
        """Policy context with structured node properties and relationship visualization"""
        try:
            if not similar_nodes:
                return "POLICY CONTEXT UNAVAILABLE\nPlease manually review full policy document"

            # Format core policy elements with full details
            core_elements = [
                "##CORE POLICY ELEMENTS:",
                *[self._format_policy_node(node) for node in similar_nodes]
            ]
            
            # Get the graph text directly from connected_data
            policy_graph = connected_data.get("graph_text", [])
            if policy_graph:
                policy_graph = "\n".join(policy_graph)
            else:
                policy_graph = "No policy relationships found"
            
            # Build final context
            return "\n".join([
                "## CLAIM STRUCTURE:",
                claim_graph or "No claim structure extracted",
                "\n" + "\n".join(core_elements),
                "\n##POLICY CONTEXT GRAPH:",
                policy_graph
            ])

        except Exception as e:
            self._log(f"Context building error: {str(e)}", level="error")
            return "Error generating policy context"

    def _format_policy_node(self, node: Dict) -> str:
        """Format individual policy node with key properties"""
        details = [
            f"({node['type'].upper()}) {node.get('text', 'Unnamed policy element')}",
        ]
        
        if node.get('value'):
            details.append(f"- Value: {node['value']}")
        if node.get('description'):
            details.append(f"- Description: {node['description']}")
        # if node.get('context'):
        #     details.append(f"- Context: {node['context']}")
            
        return "\n".join(details)

    def ask_llm_about_claim(self, claim_text: str, context: str) -> Dict[str, Any]:
        """Ask the LLM to analyze a claim against the policy context."""
        try:
            # Build the prompt
            prompt = self._build_prompt(claim_text, context)
            
            # Build proper message structure
            messages = [
                # {
                #     "role": "system", 
                #     "content": self.system_prompt
                # },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call the LLM with proper parameters
            response_content = call_openai_with_retries(
                model=self.model,
                messages=messages
            )
            
            # Parse and validate response
            response = json.loads(response_content)
            
            # Ensure array formats
            if not isinstance(response.get("limits", []), list):
                response["limits"] = []
            if not isinstance(response.get("exclusions", []), list):
                response["exclusions"] = []
            
            # Add validation for exclusion structure
            for exclusion in response["exclusions"]:
                if "applies" not in exclusion:
                    exclusion["applies"] = False
                if "source" not in exclusion:
                    exclusion["source"] = "UNKNOWN"
            
            # Log the response
            self._log("LLM Response:", level="info", is_progress=False)
            self._log("-" * 80, level="info", is_progress=False)
            self._log(json.dumps(response, indent=2), level="info", is_progress=False)
            self._log("-" * 80, level="info", is_progress=False)
            
            return response
            
        except Exception as e:
            self._log(f"Error in ask_llm_about_claim: {str(e)}", level="error", is_progress=False)
            return {
                "coverage": False,
                "explanation": f"Error analyzing claim: {str(e)}",
                "limits": [],
                "exclusions": [],
                "description": claim_text
            }

    def _extract_claim_graph(self, claim_text: str) -> str:
        """Claim graph extraction with validation and retries."""
        if ENABLE_CLAIM_GRAPH_CACHE:
            cache_key = _get_cache_key(claim_text, prefix="claim_graph")
            if cached := _get_cached_result(cache_key, "claims"):
                return cached

        prompt = dedent(f"""
        Insurance Claim Analysis Task:
        
        Claim: {claim_text}
        
        Required Output Format:
        - Use Cypher-style syntax WITHOUT code blocks
        - Only include valid node types: {', '.join(POSSIBLE_ENTITIES)}
        - Only use relationships from: {', '.join(POSSIBLE_RELATIONSHIPS)}
        
        Example Output:
        (EVENT:"Car Breakdown")-[:TRIGGERS_COVERAGE]->(COVERAGE:"Roadside Assistance")
        (COVERAGE:"Roadside Assistance")-[:HAS_APPLICABLE_LIMIT]->(LIMIT:"£500 max")
        
        Now analyze this claim and generate the relationship graph:
        """)

        messages = [
            {
                "role": "system", 
                "content": f"You are an insurance graph extraction system. Only use: "
                            f"Nodes: {POSSIBLE_ENTITIES}, Relationships: {POSSIBLE_RELATIONSHIPS}"
            },
            {"role": "user", "content": prompt}
        ]

        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=SYSTEM_PROMPT_TEMPERATURE + (attempt * RETRY_TEMPERATURE_INCREMENT),
                    max_tokens=500
                )
                raw_output = response.choices[0].message.content.strip()
                
                # Validate and clean output
                try:
                    result = self._validate_and_clean_graph(raw_output)
                    if ENABLE_CLAIM_GRAPH_CACHE:
                        _save_to_cache(cache_key, result, "claims")
                    return result
                except ValueError as e:
                    self._log(f"Invalid graph structure: {e}", level="warning")
                
                # If invalid, add error feedback to prompt
                messages.append({
                    "role": "assistant",
                    "content": raw_output
                })
                messages.append({
                    "role": "user",
                    "content": "Please correct the output to match the required format."
                })

            except Exception as e:
                self._log(f"Attempt {attempt+1} failed: {str(e)}", level="warning")

        # Return empty string when all attempts fail
        self._log("All attempts to extract claim graph failed", level="warning")
        return ""

    def _validate_and_clean_graph(self, graph_text: str) -> str:
        """
        Validates and cleans graph output to ensure it matches the required Cypher format.
        
        Args:
            graph_text: Raw graph text to validate and clean
            
        Returns:
            Cleaned graph text if valid
            
        Raises:
            ValueError: If the graph structure is invalid
        """
        # Remove any markdown code blocks
        cleaned = re.sub(r"```(cypher)?", "", graph_text)
        
        # Basic pattern validation
        node_pattern = r"\([A-Z]+:"
        rel_pattern = r"-\[:[A-Z_]+\]->"
        full_pattern = r"\([A-Z]+:.*\)-\[:.*\]->\([A-Z]+:.*\)"
        
        # Check basic patterns exist
        if not re.search(node_pattern, cleaned) or not re.search(rel_pattern, cleaned):
            raise ValueError("Missing required node or relationship patterns")
            
        # Validate complete graph structure
        if not re.search(full_pattern, cleaned):
            raise ValueError("Invalid graph structure - must contain complete node-relationship-node patterns")
        
        # Remove any special characters except those needed for Cypher
        cleaned = re.sub(r"[^a-zA-Z0-9_():<>\"\-\[\] ]", "", cleaned).strip()
        
        return cleaned

    def _create_search_embedding(self, claim_text: str) -> str:
        """Create embedding with caching and fallback to raw text."""
        if ENABLE_EMBEDDING_CACHE:
            cache_key = _get_cache_key(claim_text, prefix="embedding")
            if cached := _get_cached_result(cache_key, "embeddings"):
                self._log("Using cached embedding", level="info")
                return cached
        
        # structured_prompt = f"""Extract from claim:
        # {claim_text}
        
        # Output as JSON with:
        # - main_event
        # - damages
        # - coverage_sought
        # - amounts_claimed
        # - policy_relevant_facts
        # - potential_triggers
        # - possible_exclusions
        # """
        
        try:
            # Try structured analysis first
            # structured_analysis = call_openai_with_retries(
            #     model=self.model,
            #     messages=[{"role": "user", "content": structured_prompt}]
            # )
            # search_text = f"CLAIM: {claim_text}\nANALYSIS: {structured_analysis}"
            search_text = claim_text
            embedding = get_embedding(search_text, model="text-embedding-3-small")
            if ENABLE_EMBEDDING_CACHE:
                _save_to_cache(cache_key, embedding, "embeddings")
            return embedding
        except Exception as e:
            self._log(f"Embedding failed, using raw text hash: {str(e)}", level="error")
            return hashlib.sha256(claim_text.encode()).digest()[:1536]

    def reason_about_claim(self, claim_description: str, return_payload: bool = False):
        """Process a claim and return the analysis result."""
        try:
            # Add initial claim logging
            self._log(f"Processing claim: {claim_description[:200]}...", level="debug")
            
            # Extract claim graph first
            self._log("Extracting claim graph...", level="info", is_progress=True)
            claim_graph = self._extract_claim_graph(claim_description)
            self._log(f"Claim graph extracted: {claim_graph}", level="debug")
            
            if not claim_graph:
                self._log("Failed to extract claim graph.", level="warning", is_progress=True)
                result = {
                    "coverage": False,
                    "explanation": "Failed to process claim structure.",
                    "limits": [],
                    "exclusions": [],
                    "description": claim_description
                }
                payload = {
                    "claim": claim_description,
                    "context_sent": {
                        "system_prompt": self.system_prompt,
                        "user_prompt": "Failed to extract claim graph.",
                        "policy_context": ""
                    },
                    "analysis_result": result
                }
                if return_payload:
                    return result, payload
                return result
            
            # Get embedding for the claim
            self._log("Creating search embedding...", level="info", is_progress=True)
            claim_embedding = self._create_search_embedding(claim_description)
            
            if not claim_embedding:
                self._log("Failed to create search embedding.", level="warning", is_progress=True)
                result = {
                    "coverage": False,
                    "explanation": "Failed to process claim text for search.",
                    "limits": [],
                    "exclusions": [],
                    "description": claim_description
                }
                payload = {
                    "claim": claim_description,
                    "context_sent": {
                        "system_prompt": self.system_prompt,
                        "user_prompt": "Failed to create search embedding.",
                        "policy_context": ""
                    },
                    "analysis_result": result
                }
                if return_payload:
                    return result, payload
                return result
            
            # Get similar nodes using embeddings
            self._log("Finding similar nodes...", level="info", is_progress=True)
            similar_nodes = self.memgraph_client.find_similar_nodes(
                claim_embedding, 
                limit=MAX_RESULTS,
                min_similarity=VECTOR_SIMILARITY_THRESHOLD,
                min_importance=MIN_IMPORTANCE_SCORE
            )
            
            if LOG_SIMILAR_NODES:
                self._log(f"Similar nodes found: {len(similar_nodes)}", level="debug")
                if similar_nodes:
                    self._log(f"First {DEBUG_FIRST_N_RESULTS} results: {similar_nodes[:DEBUG_FIRST_N_RESULTS]}", level="debug")
            
            if not similar_nodes:
                self._log("No similar nodes found.", level="warning", is_progress=True)
                result = {
                    "coverage": False,
                    "explanation": "No relevant policy information found for this claim.",
                    "limits": [],
                    "exclusions": [],
                    "description": claim_description
                }
                payload = {
                    "claim": claim_description,
                    "context_sent": {
                        "system_prompt": self.system_prompt,
                        "user_prompt": "No similar nodes found.",
                        "policy_context": ""
                    },
                    "analysis_result": result
                }
                if return_payload:
                    return result, payload
                return result
            
            self._log(f"Found {len(similar_nodes)} similar nodes.", level="info", is_progress=True)
            
            # Extract subgraph for similar nodes
            self._log("Extracting policy context...", level="info", is_progress=True)
            subgraph = self.memgraph_client.extract_subgraph_for_nodes(
                similar_nodes,
                max_depth=MAX_HOP_DEPTH,
                ignore_relationships=IGNORE_RELATIONSHIP_TYPES
            )
            
            if LOG_SUBGRAPH_EXTRACTION:
                self._log(f"Found {len(subgraph.get('graph_text', []))} elements in subgraph", level="debug")
            
            # Build context from subgraph
            context = self.build_context_text(similar_nodes, subgraph, claim_graph)
            
            if not context:
                self._log("No context could be built from the subgraph.", level="warning", is_progress=True)
                result = {
                    "coverage": False,
                    "explanation": "Failed to extract relevant policy context.",
                    "limits": [],
                    "exclusions": [],
                    "description": claim_description
                }
                payload = {
                    "claim": claim_description,
                    "context_sent": {
                        "system_prompt": self.system_prompt,
                        "user_prompt": "Failed to extract relevant policy context.",
                        "policy_context": ""
                    },
                    "analysis_result": result
                }
                if return_payload:
                    return result, payload
                return result
            
            # Get LLM analysis
            self._log("Analyzing claim against policy...", level="info", is_progress=True)
            
            # Generate the actual prompt first
            prompt = self._build_prompt(claim_description, context)
            result = self.ask_llm_about_claim(claim_description, context)
            
            # Modify payload construction to include full context
            payload = {
                "claim": claim_description,
                "context_sent": {
                    "system_prompt": self.system_prompt,
                    "user_prompt": prompt,
                    "policy_context": context
                },
                "analysis_result": result
            }
            
            if return_payload:
                return result, payload
            return result
            
        except Exception as e:
            error_result = {
                "coverage": False,
                "explanation": f"Error processing claim: {str(e)}",
                "limits": [],
                "exclusions": [],
                "description": claim_description
            }
            error_payload = {
                "claim": claim_description,
                "context_sent": {
                    "system_prompt": self.system_prompt,
                    "user_prompt": "Error occurred during processing.",
                    "policy_context": ""
                },
                "analysis_result": error_result
            }
            
            if return_payload:
                return error_result, error_payload
            return error_result