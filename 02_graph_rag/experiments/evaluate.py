import json
from datetime import datetime
import pandas as pd
from pathlib import Path

def load_ground_truth():
    """Load the ground truth claims dataset"""
    path = Path(__file__).parent.parent / "data/evaluation/claims_dataset_v2_manual.json"
    return pd.read_json(path)

def load_experiment_results(file_path):
    """Load a single experiment results file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_experiment(results, ground_truth):
    """Evaluate a single experiment against ground truth"""
    df_results = pd.DataFrame(results)
    if 'true_coverage' not in df_results.columns:
        df_results['true_coverage'] = ground_truth['coverage']
    if 'true_limit_amount' not in df_results.columns:
        df_results['true_limit_amount'] = ground_truth['limit_amount']
    
    # Coverage metrics
    coverage_accuracy = len(df_results[df_results.coverage == df_results.true_coverage]) / len(df_results)
    
    # Calculate coverage confusion matrix metrics
    true_positives = len(df_results[(df_results.coverage == True) & (df_results.true_coverage == True)])
    false_positives = len(df_results[(df_results.coverage == True) & (df_results.true_coverage == False)])
    true_negatives = len(df_results[(df_results.coverage == False) & (df_results.true_coverage == False)])
    false_negatives = len(df_results[(df_results.coverage == False) & (df_results.true_coverage == True)])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Limit metrics
    # Only evaluate limit amounts for claims where both true and predicted coverage is True
    covered_claims = df_results[(df_results.coverage == True) & (df_results.true_coverage == True)]
    
    # Extract limit amounts from structured limits array
    def get_primary_limit_amount(limits):
        if not limits:
            return None
        # Get the first limit amount if available
        return limits[0].get('amount') if isinstance(limits, list) else None

    # Convert limit amounts for comparison
    df_results['normalized_limit_amount'] = df_results.apply(
        lambda x: get_primary_limit_amount(x.get('limits', [])), axis=1
    )
    
    # Calculate limit accuracy only for covered claims
    covered_claims = df_results[(df_results.coverage == True) & (df_results.true_coverage == True)]
    limit_accuracy = len(covered_claims[covered_claims.normalized_limit_amount == covered_claims.true_limit_amount]) / len(covered_claims) if len(covered_claims) > 0 else 0
    
    # Exclusion metrics
    def has_applicable_exclusions(row):
        exclusions = row.get('exclusions', [])
        if not isinstance(exclusions, list):
            return False
        return any(
            isinstance(excl, dict) and excl.get('applies', False) 
            for excl in exclusions
        )
    
    df_results['has_exclusions'] = df_results.apply(has_applicable_exclusions, axis=1)
    exclusion_count = df_results['has_exclusions'].sum()
    
    # Add debug information
    print("\nDebug Information:")
    print("Number of rows:", len(df_results))
    print("Rows with valid limits:", df_results['normalized_limit_amount'].notna().sum())
    print("Rows with exclusions:", exclusion_count)
    
    return {
        'coverage_accuracy': coverage_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'limit_accuracy': limit_accuracy,
        'covered_claims_count': len(covered_claims),
        'claims_with_exclusions': exclusion_count
    }

def find_experiment_files():
    """Find all experiment result files in generation folders"""
    experiments_dir = Path(__file__).parent
    experiment_files = []
    
    for generation_dir in experiments_dir.glob('generation-*'):
        if generation_dir.is_dir():
            for file in generation_dir.glob('*.json'):
                experiment_files.append({
                    'generation': generation_dir.name,
                    'file_path': file,
                    'file_name': file.name
                })
    
    return experiment_files

def generate_report(evaluations):
    """Generate markdown report from evaluation results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = f"# Experiment Evaluation Report - {timestamp}\n\n"
    
    # Group evaluations by generation
    generations = {}
    for eval in evaluations:
        gen = eval['generation']
        if gen not in generations:
            generations[gen] = []
        generations[gen].append(eval)
    
    # Generate report sections by generation
    for gen, evals in sorted(generations.items()):
        report += f"## {gen}\n\n"
        report += "| Experiment | Coverage Accuracy | Precision | Recall | F1 Score | Limit Accuracy | Covered Claims | Claims with Exclusions |\n"
        report += "|------------|-------------------|-----------|---------|-----------|----------------|----------------|---------------------|\n"
        
        for eval in evals:
            metrics = eval['metrics']
            report += (f"| {eval['file_name']} | {metrics['coverage_accuracy']:.3f} | "
                    f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                    f"{metrics['f1']:.3f} | {metrics['limit_accuracy']:.3f} | "
                    f"{metrics['covered_claims_count']} | {metrics['claims_with_exclusions']} |\n")
        
        report += "\n"
        
        # Add confusion matrix section
        report += "### Coverage Confusion Matrices\n\n"
        for eval in evals:
            metrics = eval['metrics']
            report += f"**{eval['file_name']}**\n\n"
            report += "```\n"
            report += "              Predicted\n"
            report += "             Pos    Neg\n"
            report += f"Actual Pos   {metrics['true_positives']}      {metrics['false_negatives']}\n"
            report += f"       Neg   {metrics['false_positives']}      {metrics['true_negatives']}\n"
            report += "```\n\n"
    
    return report

def main():
    # Load ground truth
    ground_truth = load_ground_truth()
    
    # Find all experiment files
    experiment_files = find_experiment_files()
    
    # Evaluate each experiment
    evaluations = []
    for exp in experiment_files:
        results = load_experiment_results(exp['file_path'])
        metrics = evaluate_experiment(results, ground_truth)
        evaluations.append({
            'generation': exp['generation'],
            'file_name': exp['file_name'],
            'metrics': metrics
        })
    
    # Generate and save report with timestamp
    report = generate_report(evaluations)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path(__file__).parent / f"evaluation_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Evaluation report generated: {report_path}")

if __name__ == "__main__":
    main() 