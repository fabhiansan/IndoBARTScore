#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for evaluating English text summarization using BARTScore.
This script validates our approach by using the original BARTScore implementation.
"""

import sys
import os
import torch
import argparse

# Add path to BARTScore
sys.path.append(os.path.join(os.path.dirname(__file__), 'BARTScore'))
from bart_score import BARTScorer


def evaluate_english_summarization(source_texts, summaries, batch_size=4):
    """
    Evaluate English summaries using BARTScore.
    
    Args:
        source_texts (list): List of source texts
        summaries (list): List of summaries
        batch_size (int): Batch size for processing
        
    Returns:
        dict: Evaluation results with different metrics
    """
    # Initialize the scorer with the BART-large-cnn checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Loading BART model from facebook/bart-large-cnn...")
    scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    print("BART model loaded successfully!")
    
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
    parser = argparse.ArgumentParser(description="Evaluate English summarization using BARTScore")
    parser.add_argument("--source", "-s", type=str, help="Source text file with one text per line")
    parser.add_argument("--summary", "-r", type=str, help="Summary text file with one text per line")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--use_examples", "-e", action="store_true", help="Use example texts instead of files")
    args = parser.parse_args()
    
    # Use example texts or read from files
    if args.use_examples or (not args.source and not args.summary):
        print("Using example English texts...")
        english_sources = [
            "The economy of the United States has been growing significantly in recent years. Various industrial sectors are developing rapidly, especially in technology and tourism. Foreign investment continues to increase along with stable political and security conditions. The government has made various efforts to improve the investment climate, such as simplifying licensing and providing tax incentives. Nevertheless, there are still several challenges faced, such as uneven infrastructure and economic disparities between regions.",
            
            "New York is the largest city in the United States and serves as a major center for finance and culture. The city faces various urban problems such as traffic congestion, high living costs, and air pollution. The government has implemented various policies to address these issues, including the development of mass transportation and stricter emission regulations. However, the continuously growing population and high urbanization make these problems difficult to fully resolve."
        ]

        english_summaries = [
            "The US economy is experiencing significant growth with development in the technology and tourism sectors and increased foreign investment, but still faces challenges in infrastructure and economic inequality.",
            
            "New York, as the largest US city, faces problems with traffic, high living costs, and pollution, which the government addresses with mass transit and emission regulations, but high urbanization complicates solutions."
        ]
    else:
        # Read from files
        try:
            with open(args.source, 'r', encoding='utf-8') as f:
                english_sources = [line.strip() for line in f if line.strip()]
                
            with open(args.summary, 'r', encoding='utf-8') as f:
                english_summaries = [line.strip() for line in f if line.strip()]
                
            if len(english_sources) != len(english_summaries):
                print("Error: Number of source texts and summaries must match.")
                return
        except Exception as e:
            print(f"Error reading files: {e}")
            return
    
    print("=" * 60)
    print("ENGLISH SUMMARIZATION EVALUATION WITH BARTSCORE")
    print("=" * 60)
    
    # Run evaluation
    results = evaluate_english_summarization(
        source_texts=english_sources,
        summaries=english_summaries,
        batch_size=args.batch_size
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 60)
    
    for i, (src, summary) in enumerate(zip(english_sources, english_summaries)):
        print(f"Example {i+1}:")
        # Truncate long texts for display
        disp_src = (src[:100] + '...') if len(src) > 100 else src
        disp_sum = (summary[:100] + '...') if len(summary) > 100 else summary
        
        print(f"Source: {disp_src}")
        print(f"Summary: {disp_sum}")
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
