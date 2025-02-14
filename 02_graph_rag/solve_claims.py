import os
import logging
import argparse
import json
import sys
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import URI, AUTH, DATABASE_
from database.memgraph_client import MemgraphClient
from retriever.retriever import PolicyRetriever
from experiments.evaluate import load_ground_truth, evaluate_experiment

def setup_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
    else:
        # Only show WARNING and above for most modules
        logging.basicConfig(
            level=logging.WARNING,
            format='%(message)s',
            stream=sys.stdout
        )
        # Keep INFO level for specific progress updates
        progress_logger = logging.getLogger('progress')
        progress_logger.setLevel(logging.INFO)

# Create a progress-specific logger
progress_logger = logging.getLogger('progress')

def get_model_name():
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    if provider in ['openai', 'azure']:
        return os.getenv('OPENAI_MODEL', 'default')
    elif provider == 'deepseek':
        return os.getenv('DEEPSEEK_MODEL', 'default')
    return 'default'

def sanitize_for_json(obj, visited=None):
    """Recursively sanitize an object for JSON serialization, handling circular references."""
    if visited is None:
        visited = set()

    # Get object's id to track circular references
    obj_id = id(obj)
    if obj_id in visited:
        return "<circular reference>"
    
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    visited.add(obj_id)
    
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v, visited.copy()) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item, visited.copy()) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return sanitize_for_json(obj.__dict__, visited.copy())
    else:
        return str(obj)

def save_result(result: dict, output_dir: Path, index: int):
    """Save individual result to a file."""
    filename = f"claim_{index:04d}.json"
    # Sanitize the result before saving
    sanitized_result = sanitize_for_json(result)
    with open(output_dir / filename, "w") as f:
        json.dump(sanitized_result, f, indent=4)

def load_results_up_to(output_dir: Path, n: int) -> list:
    """Load first n results from the output directory."""
    results = []
    for i in range(n):
        filename = f"claim_{i:04d}.json"
        try:
            with open(output_dir / filename) as f:
                results.append(json.load(f))
        except FileNotFoundError:
            break
    return results

def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def run_evaluation(results: list, ground_truth, n: int):
    """Run evaluation on the first n cases."""
    gt_subset = ground_truth.iloc[:n]
    results_subset = results[:n]
    
    metrics = evaluate_experiment(results_subset, gt_subset)
    
    # Convert metrics to serializable types
    metrics = {k: convert_to_serializable(v) for k, v in metrics.items()}
    
    # Create a formatted table for metrics
    table = f"""
╔══════════════════════════════════════════════════════════╗
  Progressive Evaluation Results (n={n})                    
╚══════════════════════════════════════════════════════════╝
  Coverage Accuracy   │ {metrics['coverage_accuracy']:.3f} 
  Precision           │ {metrics['precision']:.3f}         
  Recall              │ {metrics['recall']:.3f}            
  F1 Score            │ {metrics['f1']:.3f}                
  Limit Accuracy      │ {metrics['limit_accuracy']:.3f}    
  Covered Claims      │ {metrics['covered_claims_count']}  
  Claims w/Exclusions │ {metrics['claims_with_exclusions']}
╚══════════════════════════════════════════════════════════╝
"""
    progress_logger.info(table)
    
    return metrics

def load_last_processed_index(output_dir: Path) -> int:
    """Find the index of the last processed claim."""
    files = list(output_dir.glob("claim_*.json"))
    if not files:
        return -1
    indices = [int(f.stem.split('_')[1]) for f in files]
    return max(indices)

def main(claims_path: str, resume: bool = False, verbose: bool = False):
    # Setup logging based on verbosity
    setup_logging(verbose)
    
    # Initialize Memgraph client
    mg_client = MemgraphClient(uri=URI, auth=AUTH, database=DATABASE_)
    
    # Initialize retriever with verbose flag
    retriever = PolicyRetriever(mg_client, verbose=verbose)

    # Get model name from environment variables
    model = get_model_name()
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    model_identifier = f"{provider}_{model}"

    # Create or find output directory
    if resume:
        # Find the most recent matching directory
        dirs = sorted(Path("experiments/generation-5").glob(f"run_*_{model_identifier}"), reverse=True)
        if not dirs:
            progress_logger.error("No previous run found to resume")
            return
        output_dir = dirs[0]
        progress_logger.info(f"Resuming from directory: {output_dir}")
    else:
        # Create new directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiments/generation-1/run_{timestamp}_{model_identifier}")
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_logger.info(f"Created new experiment directory: {output_dir}")

    # Load claims directly from provided path
    with open(claims_path) as f:
        claims = json.load(f)
    
    # Load ground truth from the same directory as claims file
    ground_truth_path = Path(claims_path).parent / "claims_dataset_v2_manual.json"
    try:
        ground_truth = pd.read_json(ground_truth_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        progress_logger.warning(f"Ground truth file not found or invalid: {ground_truth_path}")
        ground_truth = None

    # Find starting point
    start_idx = load_last_processed_index(output_dir) + 1 if resume else 0
    
    # Track evaluation metrics over time
    evaluation_metrics = []
    
    # Load existing metrics if resuming
    metrics_file = output_dir / "progressive_metrics.json"
    if resume and metrics_file.exists():
        try:
            with open(metrics_file) as f:
                evaluation_metrics = json.load(f)
            progress_logger.info(f"Loaded {len(evaluation_metrics)} existing metrics")
        except json.JSONDecodeError as e:
            progress_logger.warning(f"Error loading metrics file: {e}")
            progress_logger.warning("Starting with empty metrics list")
            # Backup the corrupted file
            if metrics_file.exists():
                backup_file = metrics_file.with_suffix('.json.bak')
                metrics_file.rename(backup_file)
                progress_logger.info(f"Backed up corrupted metrics file to {backup_file}")
    
    # Initialize progress bar
    pbar = tqdm(
        total=len(claims),
        initial=start_idx,
        desc="Processing claims",
        disable=False  # Always show progress bar
    )
    
    try:
        for i, claim in enumerate(claims[start_idx:], start=start_idx):
            try:
                # Process the claim
                result, payload = retriever.reason_about_claim(claim["description"], return_payload=True)
                result["description"] = claim["description"]
                result["llm_payload"] = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model_identifier,
                    "input_content": payload
                }
                
                # Save individual result
                save_result(result, output_dir, i)
                
                # Update progress bar
                pbar.update(1)
                
                # Run progressive evaluation every 10 claims or at the end
                if (i + 1) % 10 == 0 or i == len(claims) - 1:
                    results = load_results_up_to(output_dir, i + 1)
                    metrics = run_evaluation(results, ground_truth, i + 1)
                    
                    # Save evaluation metrics
                    metrics['n_claims'] = i + 1
                    metrics['timestamp'] = datetime.now().isoformat()
                    evaluation_metrics.append(metrics)
                    
                    # Save metrics after each evaluation
                    try:
                        with open(metrics_file, "w") as f:
                            json.dump(evaluation_metrics, f, indent=4)
                    except Exception as e:
                        progress_logger.error(f"Error saving metrics: {e}")
                
            except KeyboardInterrupt:
                progress_logger.info("\nKeyboard interrupt detected. Saving current progress...")
                # Save final evaluation metrics
                try:
                    with open(metrics_file, "w") as f:
                        json.dump(evaluation_metrics, f, indent=4)
                except Exception as e:
                    progress_logger.error(f"Error saving metrics on interrupt: {e}")
                progress_logger.info(f"Progress saved. Processed {i} claims. Resume with --resume flag.")
                sys.exit(0)
                
    finally:
        # Clean up
        pbar.close()
        mg_client.close()
        progress_logger.info(f"All results and metrics saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solves claims from scenario dataset.")
    parser.add_argument("--claims_path", type=str, help="Path to the claims.")
    parser.add_argument("--resume", action="store_true", help="Resume from last run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    main(claims_path=args.claims_path, resume=args.resume, verbose=args.verbose)
