#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the IndoBARTScore benchmark on the SEACrowd/indosum dataset.
This script validates the benchmark setup without downloading the full dataset.
"""

import os
import sys
import json
import pandas as pd

# Add path to IndoBARTScore
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from IndoBARTScore.SUM.indobart_score import IndoBARTScorer


def test_indobartscore():
    """Test the IndoBARTScorer with sample Indonesian text."""
    print("Testing IndoBARTScorer with sample texts...")
    
    # Sample Indonesian document-summary pairs
    sample_data = {
        'article_id': ['test1', 'test2'],
        'source': [
            "Cuaca cerah hari ini membuat para petani dapat bekerja di sawah dengan lancar dan optimal. Mereka mulai bekerja sejak pagi dan berencana untuk menyelesaikan penanaman padi sebelum sore.",
            "Gempa bumi berkekuatan 5,6 magnitudo mengguncang wilayah Jawa Barat pada Senin pagi. Tidak ada korban jiwa yang dilaporkan, namun beberapa bangunan mengalami kerusakan ringan."
        ],
        'summary': [
            "Cuaca cerah membantu petani bekerja optimal di sawah.",
            "Gempa 5,6 magnitudo guncang Jawa Barat, tidak ada korban jiwa."
        ]
    }
    
    # Initialize the scorer
    scorer = IndoBARTScorer()
    
    # Score sample data
    print("Calculating source → summary scores...")
    src_to_sum_scores = scorer.score(sample_data['source'], sample_data['summary'])
    
    print("Calculating summary → source scores...")
    sum_to_src_scores = scorer.score(sample_data['summary'], sample_data['source'])
    
    # Calculate combined scores
    combined_scores = [(s + t)/2 for s, t in zip(src_to_sum_scores, sum_to_src_scores)]
    
    # Display results
    print("\nTest Results:")
    print("-" * 60)
    
    for i, (src, summary) in enumerate(zip(sample_data['source'], sample_data['summary'])):
        print(f"Example {i+1}:")
        print(f"Source: {src[:50]}...")
        print(f"Summary: {summary}")
        print(f"Source → Summary (Faithfulness): {src_to_sum_scores[i]:.4f}")
        print(f"Summary → Source (Relevance): {sum_to_src_scores[i]:.4f}")
        print(f"Combined Score: {combined_scores[i]:.4f}")
        print("-" * 60)
    
    # Save test results to allow examination
    test_results = {
        'article_id': sample_data['article_id'],
        'source': sample_data['source'],
        'summary': sample_data['summary'],
        'src_to_sum_score': src_to_sum_scores,
        'sum_to_src_score': sum_to_src_scores,
        'combined_score': combined_scores
    }
    
    # Save as CSV and JSON for reference
    os.makedirs('test_results', exist_ok=True)
    pd.DataFrame(test_results).to_csv('test_results/sample_scores.csv', index=False)
    
    with open('test_results/sample_scores.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'results': [
                    {
                        'article_id': test_results['article_id'][i],
                        'source': test_results['source'][i],
                        'summary': test_results['summary'][i],
                        'src_to_sum_score': src_to_sum_scores[i],
                        'sum_to_src_score': sum_to_src_scores[i],
                        'combined_score': combined_scores[i]
                    }
                    for i in range(len(test_results['article_id']))
                ],
                'average_scores': {
                    'src_to_sum': sum(src_to_sum_scores) / len(src_to_sum_scores),
                    'sum_to_src': sum(sum_to_src_scores) / len(sum_to_src_scores),
                    'combined': sum(combined_scores) / len(combined_scores)
                }
            },
            f,
            indent=2,
            ensure_ascii=False
        )
    
    print(f"\nTest results saved to test_results/")
    print("Benchmark setup verified successfully!")


def test_data_loader_mock():
    """Test the data loader with a mock implementation."""
    print("Testing data loader with mock implementation...")
    
    # Define mock implementation function
    def mock_load_dataset(split='test', max_samples=2):
        """Mock implementation of dataset loading."""
        print(f"Mock loading SEACrowd/indosum dataset (split: {split})...")
        
        # Create mock dataset
        mock_data = [
            {
                'id': 'article1',
                'document': "Presiden Indonesia menghadiri KTT ASEAN di Jakarta. Pertemuan tersebut membahas kerjasama ekonomi dan keamanan regional.",
                'summary': "Presiden Indonesia hadiri KTT ASEAN di Jakarta membahas ekonomi dan keamanan."
            },
            {
                'id': 'article2',
                'document': "Tim nasional sepak bola Indonesia berhasil mengalahkan Malaysia dengan skor 2-1 pada pertandingan persahabatan kemarin.",
                'summary': "Timnas Indonesia kalahkan Malaysia 2-1 dalam laga persahabatan."
            }
        ]
        
        print(f"Loaded {len(mock_data)} mock samples.")
        return mock_data
    
    # Create a simple mock processor
    class MockProcessor:
        def __init__(self, split='test'):
            self.split = split
            self.data = None
        
        def load_dataset(self, max_samples=None):
            return mock_load_dataset(self.split, max_samples)
        
        def preprocess(self, dataset=None, max_samples=None):
            if dataset is None:
                dataset = self.load_dataset(max_samples)
            
            # Process mock data
            sources = []
            summaries = []
            article_ids = []
            
            print("Preprocessing mock dataset...")
            for item in dataset:
                article_ids.append(item['id'])
                sources.append(item['document'])
                summaries.append(item['summary'])
            
            self.data = {
                'article_id': article_ids,
                'source': sources,
                'summary': summaries
            }
            
            print(f"Preprocessed {len(sources)} mock document-summary pairs.")
            return self.data
        
        def save_to_csv(self, output_path):
            if self.data is None:
                raise ValueError("No data to save. Call preprocess() first.")
            
            df = pd.DataFrame(self.data)
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} mock samples to {output_path}")
            return output_path
    
    # Test the mock processor
    mock_processor = MockProcessor()
    mock_data = mock_processor.preprocess()
    
    # Save mock data
    os.makedirs('test_data', exist_ok=True)
    mock_processor.save_to_csv('test_data/mock_indosum.csv')
    
    print("\nMock data saved to test_data/mock_indosum.csv")
    print("Data loader test completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IndoBARTScore benchmark setup")
    parser.add_argument("--test_scorer", action="store_true", help="Test the IndoBARTScorer")
    parser.add_argument("--test_loader", action="store_true", help="Test the data loader with mock implementation")
    parser.add_argument("--test_all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    # Default to running all tests if no specific test is selected
    if not (args.test_scorer or args.test_loader) or args.test_all:
        test_indobartscore()
        print("\n" + "=" * 60 + "\n")
        test_data_loader_mock()
    else:
        if args.test_scorer:
            test_indobartscore()
        if args.test_loader:
            test_data_loader_mock()
