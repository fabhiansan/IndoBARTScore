#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for evaluating IndoBARTScore on the SEACrowd/indosum dataset.
This establishes baseline scores for Indonesian summarization evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
from data_loader import IndoSumLoader

# Add path to IndoBARTScore
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from IndoBARTScore.SUM.indobart_score import IndoBARTScorer


def run_benchmark(data, batch_size=4, max_samples=None, output_dir='results'):
    """
    Run the IndoBARTScore benchmark on the given dataset.
    
    Args:
        data (dict): Dataset with sources and summaries
        batch_size (int): Batch size for processing
        max_samples (int): Maximum number of samples to process
        output_dir (str): Directory to save results
        
    Returns:
        dict: Benchmark results
    """
    # Initialize IndoBARTScorer
    print("Initializing IndoBARTScorer...")
    scorer = IndoBARTScorer()
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(data['source']):
        article_ids = data['article_id'][:max_samples]
        sources = data['source'][:max_samples]
        summaries = data['summary'][:max_samples]
    else:
        article_ids = data['article_id']
        sources = data['source']
        summaries = data['summary']
    
    # Run evaluation in batches
    print("Evaluating source → summary (faithfulness)...")
    src_to_sum_scores = []
    for i in tqdm(range(0, len(sources), batch_size)):
        batch_src = sources[i:i+batch_size]
        batch_sum = summaries[i:i+batch_size]
        batch_scores = scorer.score(batch_src, batch_sum, batch_size)
        src_to_sum_scores.extend(batch_scores)
    
    print("Evaluating summary → source (relevance)...")
    sum_to_src_scores = []
    for i in tqdm(range(0, len(sources), batch_size)):
        batch_src = sources[i:i+batch_size]
        batch_sum = summaries[i:i+batch_size]
        batch_scores = scorer.score(batch_sum, batch_src, batch_size)
        sum_to_src_scores.extend(batch_scores)
    
    # Calculate combined scores
    print("Calculating combined scores...")
    combined_scores = [(s2s + s2t)/2 for s2s, s2t in zip(src_to_sum_scores, sum_to_src_scores)]
    
    # Compile results
    results = {
        'article_id': article_ids,
        'source': sources,
        'summary': summaries,
        'src_to_sum_score': src_to_sum_scores,
        'sum_to_src_score': sum_to_src_scores,
        'combined_score': combined_scores
    }
    
    # Compute statistics
    stats = {
        'src_to_sum': {
            'mean': np.mean(src_to_sum_scores),
            'median': np.median(src_to_sum_scores),
            'std': np.std(src_to_sum_scores),
            'min': np.min(src_to_sum_scores),
            'max': np.max(src_to_sum_scores),
            'q1': np.percentile(src_to_sum_scores, 25),
            'q3': np.percentile(src_to_sum_scores, 75)
        },
        'sum_to_src': {
            'mean': np.mean(sum_to_src_scores),
            'median': np.median(sum_to_src_scores),
            'std': np.std(sum_to_src_scores),
            'min': np.min(sum_to_src_scores),
            'max': np.max(sum_to_src_scores),
            'q1': np.percentile(sum_to_src_scores, 25),
            'q3': np.percentile(sum_to_src_scores, 75)
        },
        'combined': {
            'mean': np.mean(combined_scores),
            'median': np.median(combined_scores),
            'std': np.std(combined_scores),
            'min': np.min(combined_scores),
            'max': np.max(combined_scores),
            'q1': np.percentile(combined_scores, 25),
            'q3': np.percentile(combined_scores, 75)
        }
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'article_id': article_ids,
        'src_to_sum_score': src_to_sum_scores,
        'sum_to_src_score': sum_to_src_scores,
        'combined_score': combined_scores
    })
    results_df.to_csv(os.path.join(output_dir, 'detailed_scores.csv'), index=False)
    
    # Save statistics
    with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Display statistics
    print("\nBenchmark Results Statistics:")
    print("----------------------------")
    print(f"Source → Summary (Faithfulness):")
    print(f"  Mean: {stats['src_to_sum']['mean']:.4f}")
    print(f"  Median: {stats['src_to_sum']['median']:.4f}")
    print(f"  Std Dev: {stats['src_to_sum']['std']:.4f}")
    print(f"  Range: [{stats['src_to_sum']['min']:.4f}, {stats['src_to_sum']['max']:.4f}]")
    
    print(f"\nSummary → Source (Relevance):")
    print(f"  Mean: {stats['sum_to_src']['mean']:.4f}")
    print(f"  Median: {stats['sum_to_src']['median']:.4f}")
    print(f"  Std Dev: {stats['sum_to_src']['std']:.4f}")
    print(f"  Range: [{stats['sum_to_src']['min']:.4f}, {stats['sum_to_src']['max']:.4f}]")
    
    print(f"\nCombined Score:")
    print(f"  Mean: {stats['combined']['mean']:.4f}")
    print(f"  Median: {stats['combined']['median']:.4f}")
    print(f"  Std Dev: {stats['combined']['std']:.4f}")
    print(f"  Range: [{stats['combined']['min']:.4f}, {stats['combined']['max']:.4f}]")
    
    return results, stats


def main():
    """
    Main function to run the benchmark.
    """
    parser = argparse.ArgumentParser(description="Benchmark IndoBARTScore on the SEACrowd/indosum dataset")
    parser.add_argument("--data_path", type=str, help="Path to preprocessed data file (CSV or JSONL)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], 
                        help="Dataset split to use if data_path is not specified")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    args = parser.parse_args()
    
    # Load data
    if args.data_path:
        # Load from specified file
        if args.data_path.endswith('.csv'):
            df = pd.read_csv(args.data_path)
            data = {
                'article_id': df['article_id'].tolist(),
                'source': df['source'].tolist(),
                'summary': df['summary'].tolist()
            }
        elif args.data_path.endswith('.jsonl'):
            data = {'article_id': [], 'source': [], 'summary': []}
            with open(args.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    data['article_id'].append(item['article_id'])
                    data['source'].append(item['source'])
                    data['summary'].append(item['summary'])
        else:
            raise ValueError("Unsupported file format. Use CSV or JSONL.")
    else:
        # Load and preprocess from the dataset
        loader = IndoSumLoader(split=args.split)
        data = loader.preprocess(max_samples=args.max_samples)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"indosum_{args.split}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run benchmark
    results, stats = run_benchmark(
        data=data,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=output_dir
    )
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
