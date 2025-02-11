import logging
import argparse
import sys
import json

from config.settings import URI, AUTH, DATABASE_
from database.memgraph_client import MemgraphClient
from retriever.retriever import PolicyRetriever

def setup_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    # Create formatters
    verbose_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    simple_formatter = logging.Formatter('%(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create and configure progress logger
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False  # Prevent propagation to root logger
    progress_handler = logging.StreamHandler(sys.stdout)
    progress_handler.setFormatter(simple_formatter)
    progress_logger.addHandler(progress_handler)
    
    if verbose:
        # Configure debug logger for verbose output
        debug_logger = logging.getLogger('debug')
        debug_logger.setLevel(logging.INFO)
        debug_logger.propagate = False  # Prevent propagation to root logger
        debug_handler = logging.StreamHandler(sys.stdout)
        debug_handler.setFormatter(verbose_formatter)
        debug_logger.addHandler(debug_handler)

# Create loggers
progress_logger = logging.getLogger('progress')
debug_logger = logging.getLogger('debug')

def main(query: str = None, verbose: bool = False):
    # Setup logging based on verbosity
    setup_logging(verbose)
    
    progress_logger.info("Starting retrieval process...")
    mg_client = None
    
    try:
        # Initialize Memgraph client
        progress_logger.info("Initializing Memgraph client...")
        mg_client = MemgraphClient(uri=URI, auth=AUTH, database=DATABASE_)
        
        # Initialize retriever with verbose flag
        progress_logger.info("Initializing PolicyRetriever...")
        retriever = PolicyRetriever(mg_client, verbose=verbose)
        
        if query is None:
            # Use default example if no query provided
            query = "I was driving with 7 passengers from Sheffield to London. The car broke down on the way and we had to take a hotel room for the night while the car was being repaired. The invoice is 400 pounds."
            progress_logger.info("Using default example query")
        
        result = retriever.reason_about_claim(query)
        print("\nLLM Answer:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Clean up
        if mg_client:
            progress_logger.info("Closing Memgraph client connection...")
            mg_client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a query for insurance claims.")
    parser.add_argument("--query", type=str, help="Input query for the claim.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    main(query=args.query, verbose=args.verbose)
