import mlflow
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from eval.metrics import EvaluationResult
from eval.baselines import BaselineResult

def setup_mlflow(experiment_name: str = "self-heal-rag", tracking_uri: Optional[str] = None):
    """Initialize MLflow with SQLite backend and local artifact store."""
    if tracking_uri is None:
        # Use SQLite backend with local artifact store
        tracking_uri = "sqlite:///mlflow.db"
        artifact_location = "./mlartifacts"
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def log_evaluation_run(
    run_name: str,
    variant_name: str,
    config: Dict[str, Any],
    results: List[BaselineResult],
    dataset_path: str,
    ground_truth_map: Dict[str, str]
):
    """Log a complete evaluation run to MLflow."""
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params({
            "variant": variant_name,
            "max_retries": config.get("max_retries", 0),
            "model": config.get("model", "llama-3.3-70b-versatile"),
            "temperature": config.get("temperature", 0.7),
            "chunk_size": config.get("chunk_size", 300),
            "top_k": config.get("top_k", 3),
            "dataset_size": len(results),
            "dataset_path": dataset_path,
        })
        
        # Calculate aggregate metrics
        total = len(results)
        hallucinated_count = sum(1 for r in results if r.hallucinated)
        coverage_count = sum(1 for r in results if r.coverage)
        citation_count = sum(1 for r in results if r.citations_present)
        pass_count = sum(1 for r in results if r.grade == "pass")
        avg_retries = sum(r.retry_count for r in results) / total if total > 0 else 0
        
        hallucination_rate = hallucinated_count / total if total > 0 else 0
        coverage_rate = coverage_count / total if total > 0 else 0
        citation_rate = citation_count / total if total > 0 else 0
        pass_rate = pass_count / total if total > 0 else 0
        
        # Log metrics
        mlflow.log_metrics({
            "hallucination_rate": hallucination_rate,
            "coverage_rate": coverage_rate,
            "citation_rate": citation_rate,
            "pass_rate": pass_rate,
            "avg_retries": avg_retries,
            "total_questions": total,
        })
        
        # Log per-category metrics if available
        categories = set()
        for r in results:
            # We don't have category in BaselineResult, skip for now
            pass
        
        # Create detailed results artifact
        results_data = []
        for r in results:
            results_data.append({
                "question": r.question,
                "answer": r.answer,
                "ground_truth": ground_truth_map.get(r.question, ""),
                "retrieved_context": r.retrieved_context,
                "retry_count": r.retry_count,
                "grade": r.grade,
                "hallucinated": r.hallucinated,
                "coverage": r.coverage,
                "citations_present": r.citations_present,
            })
        
        # Save as JSON artifact
        artifact_path = "evaluation_results.json"
        with open(artifact_path, "w") as f:
            json.dump(results_data, f, indent=2)
        mlflow.log_artifact(artifact_path)
        os.remove(artifact_path)
        
        # Save predictions for analysis
        predictions_path = "predictions.jsonl"
        with open(predictions_path, "w") as f:
            for r in results:
                f.write(json.dumps({
                    "question": r.question,
                    "predicted": r.answer,
                    "ground_truth": ground_truth_map.get(r.question, ""),
                    "hallucinated": r.hallucinated,
                    "coverage": r.coverage,
                }) + "\n")
        mlflow.log_artifact(predictions_path)
        os.remove(predictions_path)
        
        print(f"MLflow run logged: {run.info.run_id}")
        print(f"  Hallucination rate: {hallucination_rate:.2%}")
        print(f"  Coverage rate: {coverage_rate:.2%}")
        print(f"  Citation rate: {citation_rate:.2%}")
        print(f"  Pass rate: {pass_rate:.2%}")
        print(f"  Avg retries: {avg_retries:.2f}")
        
        return run.info.run_id

def compare_variants(experiment_name: str = "self-heal-rag"):
    """Compare all variants in an experiment."""
    import pandas as pd
    
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    if not runs:
        print("No runs found")
        return
    
    # Collect metrics
    data = []
    for run in runs:
        data.append({
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "unknown"),
            "variant": run.data.params.get("variant", "unknown"),
            "hallucination_rate": run.data.metrics.get("hallucination_rate", 0),
            "coverage_rate": run.data.metrics.get("coverage_rate", 0),
            "citation_rate": run.data.metrics.get("citation_rate", 0),
            "pass_rate": run.data.metrics.get("pass_rate", 0),
            "avg_retries": run.data.metrics.get("avg_retries", 0),
        })
    
    df = pd.DataFrame(data)
    print("\n=== Variant Comparison ===")
    print(df.to_string(index=False))
    
    # Save comparison
    df.to_csv("variant_comparison.csv", index=False)
    print("\nComparison saved to variant_comparison.csv")
    
    return df