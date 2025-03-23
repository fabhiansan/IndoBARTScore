#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for IndoBARTScore benchmark with IndoSum dataset.
This script runs a small-scale test to verify that the benchmark implementation works correctly.
"""

import os
import sys
import argparse

# Add necessary paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from IndoBARTScore.SUM.indobart_score import IndoBARTScorer
from IndoBARTScore.benchmark.indosum.run_benchmark import run_benchmark


def test_with_sample_data():
    """
    Test IndoBARTScore benchmark with sample data.
    """
    print("=" * 60)
    print("TESTING INDOBARTSCORE BENCHMARK")
    print("=" * 60)
    
    # Sample test data with expected scores pattern
    test_sources = [
        # Good matching example
        "Cuaca cerah hari ini membuat para petani dapat bekerja di sawah dengan lancar dan optimal.",
        
        # Contradictory example
        "Cuaca cerah hari ini membuat para petani dapat bekerja di sawah dengan lancar dan optimal.",
        
        # Complex example
        "Presiden Indonesia, Joko Widodo, meresmikan pembangunan jalan tol baru di Kalimantan yang akan menghubungkan beberapa kota penting. Proyek ini diharapkan dapat meningkatkan pertumbuhan ekonomi di wilayah tersebut dan membuka akses ke daerah-daerah terpencil. Gubernur Kalimantan menyampaikan apresiasi atas perhatian pemerintah pusat terhadap pembangunan infrastruktur di pulau tersebut."
    ]
    
    test_summaries = [
        # Good matching example
        "Cuaca yang cerah mendukung kegiatan bertani hari ini.",
        
        # Contradictory example
        "Cuaca yang mendung menghambat pekerjaan di sawah.",
        
        # Complex example with length variation
        "Presiden Jokowi meresmikan jalan tol di Kalimantan untuk mendorong ekonomi dan membuka akses daerah terpencil."
    ]
    
    print("\nRunning test with sample data...")
    print(f"Number of test examples: {len(test_sources)}")
    
    # Run benchmark on test data
    results = run_benchmark(
        sources=test_sources,
        summaries=test_summaries,
        batch_size=2,
        save_results=True,
        output_dir="./test_results"
    )
    
    # Verify that the benchmark ran successfully
    print("\nVerifying test results...")
    
    # 1. Check if all expected score types are present
    score_types = ["source_to_summary", "summary_to_source", "combined"]
    for score_type in score_types:
        assert score_type in results["scores"], f"Missing score type: {score_type}"
        print(f"✓ Found score type: {score_type}")
    
    # 2. Check if individual scores match expected pattern
    assert len(results["individual_scores"]) == 3, "Incorrect number of individual scores"
    
    # The first example (matching) should have better scores than the second (contradictory)
    first_combined = results["individual_scores"][0]["combined_score"]
    second_combined = results["individual_scores"][1]["combined_score"]
    assert first_combined > second_combined, f"First example should score better than second: {first_combined} vs {second_combined}"
    print("✓ Score pattern matches expectation: matching example scores better than contradictory example")
    
    print("\nTest completed successfully!")
    return results


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test IndoBARTScore benchmark with sample data")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Directory to save test results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run test
    test_with_sample_data()


if __name__ == "__main__":
    main()
