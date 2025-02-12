import argparse
import logging
import sys

from main import main as ingest_main
from retrieve import main as retrieve_main
from solve_claims import main as solve_claims_main

def setup_logging(verbose: bool):
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

def main():
    parser = argparse.ArgumentParser(
        description="Graph RAG CLI - A graph-based Retrieval Augmented Generation system for insurance policy analysis"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command (main.py functionality)
    ingest_parser = subparsers.add_parser('ingest', help='Ingest policy documents into the graph database')
    ingest_parser.add_argument('--verbose', action='store_true', help='Enable detailed debug logging')
    
    # Query command (retrieve.py functionality)
    query_parser = subparsers.add_parser('query', help='Query the system with a single claim')
    query_parser.add_argument('--query', type=str, help='Input query for the claim')
    query_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Batch command (solve_claims.py functionality)
    batch_parser = subparsers.add_parser('batch', help='Process multiple claims from a file')
    batch_parser.add_argument('--claims_path', type=str, help='Path to the claims file')
    batch_parser.add_argument('--resume', action='store_true', help='Resume from last run')
    batch_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging based on verbose flag
    setup_logging(getattr(args, 'verbose', False))
    
    try:
        if args.command == 'ingest':
            ingest_main()
        elif args.command == 'query':
            retrieve_main(query=args.query, verbose=args.verbose)
        elif args.command == 'batch':
            solve_claims_main(claims_path=args.claims_path, resume=args.resume, verbose=args.verbose)
    except Exception as e:
        logging.error(f"Error executing {args.command} command: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 