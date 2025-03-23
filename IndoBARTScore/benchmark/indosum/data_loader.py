#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader for IndoSum dataset to be used with IndoBARTScore benchmarking.
IndoSum dataset: https://huggingface.co/datasets/SEACrowd/indosum
"""

import os
import argparse
import random
from datasets import load_dataset, load_from_disk
from tqdm import tqdm


def load_indosum_dataset(data_path=None, split="train", sample_size=None, seed=42):
    """
    Load the IndoSum dataset either from local disk or from Hugging Face.
    
    Args:
        data_path (str, optional): Path to local dataset. If None, load from Hugging Face.
        split (str, optional): Dataset split to load (train, test, validation).
        sample_size (int, optional): Number of samples to return. If None, return all.
        seed (int, optional): Random seed for sampling.
        
    Returns:
        list: List of dictionaries with 'article' and 'summary' keys.
    """
    random.seed(seed)
    
    if data_path and os.path.exists(data_path):
        print(f"Loading IndoSum dataset from local path: {data_path}")
        try:
            dataset = load_from_disk(data_path)
            # If this is the entire dataset with splits, get the requested split
            if hasattr(dataset, split):
                dataset = dataset[split]
            # Otherwise, assume it's already the right split
        except Exception as e:
            print(f"Error loading from disk: {e}")
            print("Attempting to load from Hugging Face...")
            dataset = load_dataset("SEACrowd/indosum", split=split)
    else:
        print(f"Loading IndoSum dataset from Hugging Face, split: {split}")
        dataset = load_dataset("SEACrowd/indosum", split=split)
    
    print(f"Dataset loaded successfully with {len(dataset)} examples")
    
    # Convert to list of dictionaries
    data_list = [{"article": item["article"], "summary": item["summary"]} for item in dataset]
    
    # Sample if requested
    if sample_size and sample_size < len(data_list):
        print(f"Sampling {sample_size} examples from dataset")
        data_list = random.sample(data_list, sample_size)
    
    return data_list


def preprocess_texts(data_list, max_length=1024):
    """
    Clean and preprocess the texts for evaluation.
    
    Args:
        data_list (list): List of dictionaries with 'article' and 'summary' keys.
        max_length (int, optional): Maximum length for truncation.
        
    Returns:
        tuple: Lists of source texts and summaries.
    """
    sources = []
    summaries = []
    
    for item in tqdm(data_list, desc="Preprocessing texts"):
        # Basic cleaning
        article = item["article"].strip()
        summary = item["summary"].strip()
        
        # Truncate if needed
        if len(article) > max_length:
            article = article[:max_length]
            
        sources.append(article)
        summaries.append(summary)
    
    print(f"Preprocessing complete. {len(sources)} source-summary pairs ready.")
    return sources, summaries


def save_samples(sources, summaries, output_dir, filename="samples.txt", num_samples=5):
    """
    Save sample source-summary pairs to a file for inspection.
    
    Args:
        sources (list): List of source texts.
        summaries (list): List of summary texts.
        output_dir (str): Directory to save the file.
        filename (str, optional): Filename to save.
        num_samples (int, optional): Number of samples to save.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("IndoSum Dataset Samples\n")
        f.write("=" * 80 + "\n\n")
        
        for i in range(min(num_samples, len(sources))):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Source: {sources[i][:500]}...")
            f.write("\n\n")
            f.write(f"Summary: {summaries[i]}")
            f.write("\n\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and prepare IndoSum dataset")
    parser.add_argument("--data_path", type=str, default=None, help="Path to local dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to use")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--save_samples", action="store_true", help="Save sample data for inspection")
    args = parser.parse_args()
    
    # Load dataset
    data_list = load_indosum_dataset(args.data_path, args.split, args.sample_size)
    
    # Preprocess texts
    sources, summaries = preprocess_texts(data_list)
    
    # Save samples if requested
    if args.save_samples:
        save_samples(sources, summaries, args.output_dir)
        
    print("Data preparation complete.")
