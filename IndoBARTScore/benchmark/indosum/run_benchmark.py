#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run benchmark for IndoBARTScore using the IndoSum dataset.
This script calculates baseline scores for Indonesian summarization evaluation.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

# Add necessary paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from IndoBARTScore.SUM.indobart_score import IndoBARTScorer
from IndoBARTScore.benchmark.indosum.data_loader import load_indosum_dataset, preprocess_texts


def run_benchmark(sources, summaries, batch_size=4, save_results=True, output_dir="./results"):
    """
    Run IndoBARTScore benchmark on IndoSum dataset examples.
    
    Args:
        sources (list): List of source texts
        summaries (list): List of summary texts
        batch_size (int): Batch size for processing
        save_results (bool): Whether to save results to disk
        output_dir (str): Directory to save results
        
    Returns:
        dict: Benchmark results with statistics
    """
    # Create scorer
    print("Initializing IndoBARTScorer...")
    scorer = IndoBARTScorer()
    
    # Initialize results storage
    results = {
        "dataset": "IndoSum",
        "num_examples": len(sources),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size": batch_size,
        "scores": {}
    }
    
    # Score from source to summary (faithfulness)
    print("\nCalculating Source → Summary scores (faithfulness)...")
    start_time = time.time()
    src_to_sum_scores = scorer.score(sources, summaries, batch_size)
    faithfulness_time = time.time() - start_time
    print(f"Faithfulness scoring completed in {faithfulness_time:.2f} seconds")
    
    # Score from summary to source (relevance)
    print("\nCalculating Summary → Source scores (relevance)...")
    start_time = time.time()
    sum_to_src_scores = scorer.score(summaries, sources, batch_size)
    relevance_time = time.time() - start_time
    print(f"Relevance scoring completed in {relevance_time:.2f} seconds")
    
    # Calculate combined scores
    print("\nCalculating combined scores...")
    combined_scores = [(s2s + s2t)/2 for s2s, s2t in zip(src_to_sum_scores, sum_to_src_scores)]
    
    # Calculate statistics
    results["scores"]["source_to_summary"] = {
        "mean": np.mean(src_to_sum_scores),
        "median": np.median(src_to_sum_scores),
        "std": np.std(src_to_sum_scores),
        "min": np.min(src_to_sum_scores),
        "max": np.max(src_to_sum_scores),
        "percentile_25": np.percentile(src_to_sum_scores, 25),
        "percentile_75": np.percentile(src_to_sum_scores, 75)
    }
    
    results["scores"]["summary_to_source"] = {
        "mean": np.mean(sum_to_src_scores),
        "median": np.median(sum_to_src_scores),
        "std": np.std(sum_to_src_scores),
        "min": np.min(sum_to_src_scores),
        "max": np.max(sum_to_src_scores),
        "percentile_25": np.percentile(sum_to_src_scores, 25),
        "percentile_75": np.percentile(sum_to_src_scores, 75)
    }
    
    results["scores"]["combined"] = {
        "mean": np.mean(combined_scores),
        "median": np.median(combined_scores),
        "std": np.std(combined_scores),
        "min": np.min(combined_scores),
        "max": np.max(combined_scores),
        "percentile_25": np.percentile(combined_scores, 25),
        "percentile_75": np.percentile(combined_scores, 75)
    }
    
    results["timing"] = {
        "faithfulness_seconds": faithfulness_time,
        "relevance_seconds": relevance_time,
        "total_seconds": faithfulness_time + relevance_time,
        "examples_per_second": len(sources) / (faithfulness_time + relevance_time)
    }
    
    # Save detailed scores for potential further analysis
    individual_scores = []
    for i, (src, summ, s2s, s2t, comb) in enumerate(zip(
            sources, summaries, src_to_sum_scores, sum_to_src_scores, combined_scores)):
        individual_scores.append({
            "id": i,
            "source_text": src[:100] + "..." if len(src) > 100 else src,  # Truncate for readability
            "summary_text": summ[:100] + "..." if len(summ) > 100 else summ,
            "source_to_summary_score": s2s,
            "summary_to_source_score": s2t,
            "combined_score": comb
        })
    results["individual_scores"] = individual_scores
    
    # Save results if requested
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"indosum_benchmark_{timestamp}.json")
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {results_file}")
        
        # Also save a CSV with individual scores
        df = pd.DataFrame(individual_scores)
        csv_file = os.path.join(output_dir, f"indosum_scores_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Individual scores saved to {csv_file}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("IndoBARTScore Benchmark Results")
    print("=" * 60)
    print(f"Dataset: IndoSum (N={len(sources)})")
    print("\nSource → Summary (Faithfulness):")
    print(f"  Mean: {results['scores']['source_to_summary']['mean']:.4f}")
    print(f"  Median: {results['scores']['source_to_summary']['median']:.4f}")
    print(f"  Standard Deviation: {results['scores']['source_to_summary']['std']:.4f}")
    print(f"  Range: [{results['scores']['source_to_summary']['min']:.4f}, {results['scores']['source_to_summary']['max']:.4f}]")
    
    print("\nSummary → Source (Relevance):")
    print(f"  Mean: {results['scores']['summary_to_source']['mean']:.4f}")
    print(f"  Median: {results['scores']['summary_to_source']['median']:.4f}")
    print(f"  Standard Deviation: {results['scores']['summary_to_source']['std']:.4f}")
    print(f"  Range: [{results['scores']['summary_to_source']['min']:.4f}, {results['scores']['summary_to_source']['max']:.4f}]")
    
    print("\nCombined Score:")
    print(f"  Mean: {results['scores']['combined']['mean']:.4f}")
    print(f"  Median: {results['scores']['combined']['median']:.4f}")
    print(f"  Standard Deviation: {results['scores']['combined']['std']:.4f}")
    print(f"  Range: [{results['scores']['combined']['min']:.4f}, {results['scores']['combined']['max']:.4f}]")
    
    print("\nTiming:")
    print(f"  Total Processing Time: {results['timing']['total_seconds']:.2f} seconds")
    print(f"  Examples Per Second: {results['timing']['examples_per_second']:.2f}")
    print("=" * 60)
    
    return results


def main():
    """Main function to run the benchmark script."""
    parser = argparse.ArgumentParser(description="Run IndoBARTScore benchmark on IndoSum dataset")
    parser.add_argument("--data_path", type=str, default=None, 
                        help="Path to local IndoSum dataset (if None, load from Hugging Face)")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use (train, test, validation)")
    parser.add_argument("--sample_size", type=int, default=500, 
                        help="Number of samples to evaluate (use smaller value for testing)")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for processing")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    print("=" * 60)
    print("IndoBARTScore Benchmark for IndoSum Dataset")
    print("=" * 60)
    
    # Load dataset
    data_list = load_indosum_dataset(
        data_path=args.data_path,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed
    )
    
    # Preprocess texts
    sources, summaries = preprocess_texts(data_list)
    
    # Run benchmark
    run_benchmark(
        sources=sources,
        summaries=summaries,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
