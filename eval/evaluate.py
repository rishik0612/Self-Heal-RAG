import argparse
import json
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import load_dataset, HallucinationDetector, CoverageCalculator, CitationChecker
from eval.baselines import get_all_baselines, run_baseline, BaselineResult
from eval.mlflow_tracking import setup_mlflow, log_evaluation_run, compare_variants
from eval.stats import mcnemar_test, bootstrap_diff_ci

def run_evaluation(
    dataset_path: str = "eval/dataset.jsonl",
    variants: List[str] = None,
    experiment_name: str = "self-heal-rag",
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_retries: int = 2,
    limit: int = None,
):
    """Run evaluation across all baselines."""
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]
        print(f"Running limited subset of {len(dataset)} questions (--limit)")
    print(f"Loaded {len(dataset)} questions")
    
    ground_truth_map = {item["question"]: item["ground_truth"] for item in dataset}
    
    # Initialize evaluators
    print("Initializing evaluators...")
    hallucination_detector = HallucinationDetector(model_name=model, temperature=temperature)
    coverage_calculator = CoverageCalculator(model_name=model, temperature=temperature)
    citation_checker = CitationChecker()
    
    # Setup MLflow
    print("Setting up MLflow...")
    setup_mlflow(experiment_name)
    
    # Get baselines
    all_baselines = get_all_baselines()
    if variants:
        baselines = [(name, graph) for name, graph in all_baselines if name in variants]
    else:
        baselines = all_baselines
    
    print(f"Running {len(baselines)} baseline variants...")
    
    config = {
        "model": model,
        "temperature": temperature,
        "max_retries": max_retries,
        "chunk_size": 300,
        "top_k": 3,
    }
    
    all_results = {}
    
    for variant_name, graph in baselines:
        print(f"\n=== Running {variant_name} ===")
        results = []
        
        for i, item in enumerate(dataset):
            question = item["question"]
            print(f"  [{i+1}/{len(dataset)}] {question[:60]}...")
            
            result = run_baseline(
                graph, question, variant_name,
                hallucination_detector, coverage_calculator, citation_checker
            )
            results.append(result)
        
        all_results[variant_name] = results
        
        # Log to MLflow
        run_name = f"{variant_name}_{os.path.basename(dataset_path).replace('.jsonl', '')}"
        log_evaluation_run(
            run_name=run_name,
            variant_name=variant_name,
            config=config,
            results=results,
            dataset_path=dataset_path,
            ground_truth_map=ground_truth_map
        )
    
    # Compare variants
    print("\n=== Comparing Variants ===")
    compare_variants(experiment_name)

    run_pairwise_stats(all_results)

    return all_results


def _as_records(results: List[BaselineResult]) -> List[Dict]:
    return [
        {
            "question": r.question,
            "hallucinated": bool(r.hallucinated),
            "coverage": bool(r.coverage),
        }
        for r in results
    ]


def run_pairwise_stats(all_results: Dict[str, List[BaselineResult]]):
    """Paired statistical comparison between every pair of variants."""
    variants = list(all_results.keys())
    if len(variants) < 2:
        return

    print("\n=== Paired Significance Tests (matched per-question) ===")
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a, b = variants[i], variants[j]
            base_records = _as_records(all_results[a])
            var_records = _as_records(all_results[b])
            for metric in ("hallucinated", "coverage"):
                base_by_q = {r["question"]: bool(r[metric]) for r in base_records}
                var_by_q = {r["question"]: bool(r[metric]) for r in var_records}
                pairs = [(base_by_q[q], var_by_q[q]) for q in base_by_q if q in var_by_q]
                if not pairs:
                    continue
                mn = mcnemar_test(pairs)
                boot = bootstrap_diff_ci(pairs)
                direction = "lower is better" if metric == "hallucinated" else "higher is better"
                print(f"\n[{a} -> {b}] metric={metric} ({direction})  n={mn['n']}")
                print(f"  {a}: {mn['baseline_rate']:.2%}  {b}: {mn['variant_rate']:.2%}  delta={mn['delta']:+.2%}")
                print(f"  McNemar p={mn['p_value']:.4f}  "
                      f"bootstrap CI [{boot['ci_low']:+.2%}, {boot['ci_high']:+.2%}] "
                      f"(excludes 0: {boot['ci_excludes_zero']})")
                verdict = "SIGNIFICANT" if (mn['p_value'] < 0.05 and boot['ci_excludes_zero']) else "not significant"
                print(f"  => {verdict} at alpha=0.05")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Self-Healing RAG variants")
    parser.add_argument("--dataset", default="eval/dataset.jsonl", help="Path to evaluation dataset")
    parser.add_argument("--variants", nargs="+", choices=["single_pass", "one_retry", "two_retry"], 
                        help="Specific variants to run (default: all)")
    parser.add_argument("--experiment", default="self-heal-rag", help="MLflow experiment name")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq model for evaluation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for evaluation LLM")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions")
    parser.add_argument("--compare-only", action="store_true", help="Only compare existing MLflow runs")
    
    args = parser.parse_args()
    
    if args.compare_only:
        compare_variants(args.experiment)
        return
    
    run_evaluation(
        dataset_path=args.dataset,
        variants=args.variants,
        experiment_name=args.experiment,
        model=args.model,
        temperature=args.temperature,
        limit=args.limit,
    )

if __name__ == "__main__":
    main()