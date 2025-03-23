#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for evaluating Indonesian text summarization using IndoBARTScore.
Based on the structure in the BARTScore/SUM folder.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from IndoBARTScore.SUM.indobart_score import IndoBARTScorer


def evaluate_indonesian_summarization(source_texts, summaries, batch_size=4):
    """
    Evaluate Indonesian summaries using IndoBARTScore.
    
    Args:
        source_texts (list): List of source texts
        summaries (list): List of summaries
        batch_size (int): Batch size for processing
        
    Returns:
        dict: Evaluation results with different metrics
    """
    # Initialize the scorer
    scorer = IndoBARTScorer()
    
    # Score from source to summary (tests faithfulness)
    print("Calculating source → summary scores (faithfulness)...")
    src_to_sum_scores = scorer.score(source_texts, summaries, batch_size)
    
    # Score from summary to source (tests relevance)
    print("Calculating summary → source scores (relevance)...")
    sum_to_src_scores = scorer.score(summaries, source_texts, batch_size)
    
    # Calculate average score (combines both directions)
    print("Calculating combined scores...")
    avg_scores = [(s2s + s2t)/2 for s2s, s2t in zip(src_to_sum_scores, sum_to_src_scores)]
    
    return {
        "src_to_sum": src_to_sum_scores,  # Faithfulness: does the summary capture the source
        "sum_to_src": sum_to_src_scores,  # Relevance: is the summary content in the source
        "avg_score": avg_scores           # Combined score
    }


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Evaluate Indonesian summarization using IndoBARTScore")
    parser.add_argument("--source", "-s", type=str, help="Source text file with one text per line")
    parser.add_argument("--summary", "-r", type=str, help="Summary text file with one text per line")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--use_examples", "-e", action="store_true", help="Use example texts instead of files")
    args = parser.parse_args()
    
    # Use example texts or read from files
    if args.use_examples or (not args.source and not args.summary):
        print("Using example Indonesian texts...")
        indonesian_sources = [
            "Cuaca cerah hari ini membuat para petani dapat bekerja di sawah dengan lancar dan optimal",
            "Cuaca cerah hari ini membuat para petani dapat bekerja di sawah dengan lancar dan optimal."
        ]

        indonesian_summaries = [
            "Cuaca yang cerah mendukung kegiatan bertani hari ini.",
            "Cuaca yang mendung menghambat pekerjaan di sawah."
        ]
    else:
        # Read from files
        try:
            with open(args.source, 'r', encoding='utf-8') as f:
                indonesian_sources = [line.strip() for line in f if line.strip()]
                
            with open(args.summary, 'r', encoding='utf-8') as f:
                indonesian_summaries = [line.strip() for line in f if line.strip()]
                
            if len(indonesian_sources) != len(indonesian_summaries):
                print("Error: Number of source texts and summaries must match.")
                return
        except Exception as e:
            print(f"Error reading files: {e}")
            return
    
    print("=" * 60)
    print("INDONESIAN SUMMARIZATION EVALUATION WITH INDOBARTSCORE")
    print("=" * 60)
    
    # Run evaluation
    results = evaluate_indonesian_summarization(
        source_texts=indonesian_sources,
        summaries=indonesian_summaries,
        batch_size=args.batch_size
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 60)
    
    for i, (src, summary) in enumerate(zip(indonesian_sources, indonesian_summaries)):
        print(f"Example {i+1}:")
        print(f"Source: {src}")
        print(f"Summary: {summary}")
        print(f"Source → Summary (Faithfulness): {results['src_to_sum'][i]:.4f}")
        print(f"Summary → Source (Relevance): {results['sum_to_src'][i]:.4f}")
        print(f"Combined Score: {results['avg_score'][i]:.4f}")
        print("-" * 60)
    
    print("\nInterpretation:")
    print("- Higher scores (closer to 0) indicate better quality summaries")
    print("- Faithfulness measures how well the summary captures source content")
    print("- Relevance measures how well the summary stays on topic")
    print("- Combined score balances both metrics")


if __name__ == "__main__":
    main()
