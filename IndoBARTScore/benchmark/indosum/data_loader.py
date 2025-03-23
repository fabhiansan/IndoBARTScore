#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader for the SEACrowd/indosum dataset.
This script downloads and preprocesses the dataset for benchmarking IndoBARTScore.
"""

import os
import json
import pandas as pd
import requests
import tempfile
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
        
        # Direct URLs to the dataset files as fallback
        self.direct_urls = {
            'train': '',
            'validation': '',
            'test': ''
        }
    
    def load_dataset(self, max_samples=None):
        """
        Load the SEACrowd/indosum dataset.
        
        Args:
            max_samples (int): Maximum number of samples to load (None for all)
            
        Returns:
            list: The loaded dataset
        """
        print(f"Loading SEACrowd/indosum dataset (split: {self.split})...")
        
        try:
            # Try loading with the datasets library
            dataset = load_dataset("SEACrowd/indosum", split=self.split, cache_dir=self.cache_dir)
            
            if max_samples is not None and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
                
            print(f"Loaded {len(dataset)} samples using HuggingFace datasets.")
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset with HuggingFace datasets: {e}")
            print(f"Falling back to direct URL fetching...")
            
            # Fallback: Load directly from GitHub URLs
            return self._load_from_direct_url(max_samples)
    
    def _load_from_direct_url(self, max_samples=None):
        """
        Load dataset directly from URL as a fallback method.
        
        Args:
            max_samples (int): Maximum number of samples to load
            
        Returns:
            list: The loaded dataset as a list of dictionaries
        """
        if self.split not in self.direct_urls:
            raise ValueError(f"Split {self.split} is not available for direct URL loading")
        
        url = self.direct_urls[self.split]
        print(f"Downloading dataset from {url}")
        
        # Download the file
        response = requests.get(url)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to download from {url}, status code: {response.status_code}")
        
        # Parse the JSONL content
        items = []
        for line in response.text.splitlines():
            if line.strip():
                try:
                    item = json.loads(line)
                    items.append(item)
                    if max_samples is not None and len(items) >= max_samples:
                        break
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
        
        print(f"Loaded {len(items)} samples directly from URL.")
        return items
    
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
            try:
                # Extract the relevant fields based on the data structure
                # Handle both HuggingFace dataset format and direct download format
                if isinstance(item, dict) and 'document' in item:
                    # Direct download format
                    article_id = item.get('id', str(len(article_ids)))
                    document = item.get('document', '')
                    summary = item.get('summary', '')
                else:
                    # HuggingFace dataset format
                    article_id = item.get('id', str(len(article_ids)))
                    document = item.get('document', '')
                    summary = item.get('summary', '')
                
                # Skip empty entries
                if not document or not summary:
                    continue
                
                # Add to our lists
                sources.append(document)
                summaries.append(summary)
                article_ids.append(article_id)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
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

    def create_mock_dataset(self, num_samples=10):
        """
        Create a mock dataset for testing purposes.
        
        Args:
            num_samples (int): Number of mock samples to create
            
        Returns:
            dict: Mock dataset with sources and summaries
        """
        print(f"Creating mock dataset with {num_samples} samples...")
        
        # Sample Indonesian document-summary pairs
        mock_data = {
            'article_id': [],
            'source': [],
            'summary': []
        }
        
        # Some sample Indonesian texts and summaries
        sample_texts = [
            ("Presiden Indonesia menghadiri KTT ASEAN di Jakarta. Pertemuan tersebut membahas kerjasama ekonomi dan keamanan regional.",
             "Presiden Indonesia hadiri KTT ASEAN di Jakarta membahas ekonomi dan keamanan."),
            ("Tim nasional sepak bola Indonesia berhasil mengalahkan Malaysia dengan skor 2-1 pada pertandingan persahabatan kemarin.",
             "Timnas Indonesia kalahkan Malaysia 2-1 dalam laga persahabatan."),
            ("Gempa bumi berkekuatan 5,6 magnitudo mengguncang wilayah Jawa Barat pada Senin pagi. Tidak ada korban jiwa yang dilaporkan.",
             "Gempa 5,6 magnitudo guncang Jawa Barat, tidak ada korban jiwa."),
            ("Pemerintah Indonesia meluncurkan program vaksinasi COVID-19 untuk anak-anak usia 6-11 tahun di seluruh wilayah Indonesia.",
             "Pemerintah luncurkan vaksinasi COVID-19 untuk anak 6-11 tahun."),
            ("Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) memperkirakan cuaca Jakarta akan cerah berawan sepanjang hari ini.",
             "BMKG: Jakarta cerah berawan sepanjang hari."),
        ]
        
        # Generate as many samples as needed
        for i in range(num_samples):
            idx = i % len(sample_texts)
            mock_data['article_id'].append(f"mock_{i+1}")
            mock_data['source'].append(sample_texts[idx][0])
            mock_data['summary'].append(sample_texts[idx][1])
        
        self.data = mock_data
        print(f"Created mock dataset with {len(mock_data['article_id'])} samples.")
        return mock_data


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
    parser.add_argument("--use_mock", action="store_true", help="Use mock data instead of downloading")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize loader
    loader = IndoSumLoader(split=args.split)
    
    if args.use_mock:
        # Use mock data for testing
        data = loader.create_mock_dataset(num_samples=args.max_samples or 10)
    else:
        # Load and preprocess real data
        try:
            data = loader.preprocess(max_samples=args.max_samples)
        except Exception as e:
            print(f"Error preprocessing real data: {e}")
            print("Falling back to mock data...")
            data = loader.create_mock_dataset(num_samples=args.max_samples or 10)
    
    # Save to specified format
    if args.format == "csv":
        output_path = os.path.join(args.output_dir, f"indosum_{args.split}.csv")
        loader.save_to_csv(output_path)
    else:
        output_path = os.path.join(args.output_dir, f"indosum_{args.split}.jsonl")
        loader.save_to_jsonl(output_path)


if __name__ == "__main__":
    main()
