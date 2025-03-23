#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader for the SEACrowd/indosum dataset.
This script downloads and preprocesses the dataset for benchmarking IndoBARTScore.
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


class IndoSumLoader:
    """
    Loader for the SEACrowd/indosum dataset for benchmarking IndoBARTScore.
    """
    
    def __init__(self, cache_dir=None, split='train'):
        """
        Initialize the IndoSumLoader.
        
        Args:
            cache_dir (str): Directory to cache the dataset
            split (str): Dataset split to load ('train', 'validation', or 'test')
        """
        self.cache_dir = cache_dir
        self.split = split
        self.data = None
    
    def load_dataset(self, max_samples=None):
        """
        Load the SEACrowd/indosum dataset.
        
        Args:
            max_samples (int): Maximum number of samples to load (None for all)
            
        Returns:
            dict: The loaded dataset
        """
        print(f"Loading SEACrowd/indosum dataset (split: {self.split})...")
        dataset = load_dataset("SEACrowd/indosum", split=self.split, cache_dir=self.cache_dir)
        
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            
        print(f"Loaded {len(dataset)} samples.")
        return dataset
    
    def preprocess(self, dataset=None, max_samples=None):
        """
        Preprocess the dataset for benchmarking.
        
        Args:
            dataset: Loaded dataset (if None, will load the dataset)
            max_samples (int): Maximum number of samples to process
            
        Returns:
            dict: Processed dataset with sources and summaries
        """
        if dataset is None:
            dataset = self.load_dataset(max_samples)
        
        # Extract document and summary pairs
        sources = []
        summaries = []
        article_ids = []
        
        print("Preprocessing dataset...")
        for item in tqdm(dataset):
            # Extract the relevant fields
            article_id = item.get('id', '')
            document = item.get('document', '')
            summary = item.get('summary', '')
            
            # Skip empty entries
            if not document or not summary:
                continue
            
            # Add to our lists
            sources.append(document)
            summaries.append(summary)
            article_ids.append(article_id)
        
        self.data = {
            'article_id': article_ids,
            'source': sources,
            'summary': summaries
        }
        
        print(f"Preprocessed {len(sources)} valid document-summary pairs.")
        return self.data
    
    def save_to_csv(self, output_path):
        """
        Save the preprocessed data to a CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
            
        Returns:
            str: Path to the saved CSV file
        """
        if self.data is None:
            raise ValueError("No data to save. Call preprocess() first.")
        
        df = pd.DataFrame(self.data)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} samples to {output_path}")
        return output_path
    
    def save_to_jsonl(self, output_path):
        """
        Save the preprocessed data to a JSONL file.
        
        Args:
            output_path (str): Path to save the JSONL file
            
        Returns:
            str: Path to the saved JSONL file
        """
        if self.data is None:
            raise ValueError("No data to save. Call preprocess() first.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(len(self.data['article_id'])):
                entry = {
                    'article_id': self.data['article_id'][i],
                    'source': self.data['source'][i],
                    'summary': self.data['summary'][i]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(self.data['article_id'])} samples to {output_path}")
        return output_path


def main():
    """
    Main function to demonstrate usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and preprocess the SEACrowd/indosum dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save processed data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], 
                        help="Dataset split to load")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to load")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "jsonl"], 
                        help="Output file format")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize loader and preprocess data
    loader = IndoSumLoader(split=args.split)
    data = loader.preprocess(max_samples=args.max_samples)
    
    # Save to specified format
    if args.format == "csv":
        output_path = os.path.join(args.output_dir, f"indosum_{args.split}.csv")
        loader.save_to_csv(output_path)
    else:
        output_path = os.path.join(args.output_dir, f"indosum_{args.split}.jsonl")
        loader.save_to_jsonl(output_path)


if __name__ == "__main__":
    main()
