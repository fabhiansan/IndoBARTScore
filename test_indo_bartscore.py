#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for evaluating Indonesian text summarization using BARTScore with IndoBART.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import MBartForConditionalGeneration

# Add paths to required modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'indobenchmark-toolkit/src'))
from indobenchmark.tokenization_indonlg import IndoNLGTokenizer


class IndoBARTScorer:
    """BARTScore implementation that uses IndoBART model for Indonesian text."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', max_length=1024, checkpoint='indobenchmark/indobart-v2'):
        # Set up model
        self.device = device
        self.max_length = max_length
        print(f"Using device: {device}")
        print(f"Loading IndoBART model from {checkpoint}...")
        
        self.tokenizer = IndoNLGTokenizer.from_pretrained(checkpoint, additional_special_tokens=[])
        self.model = MBartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        
        # Language token for Indonesian
        self.lang_token = '[indonesian]'

        # Set up loss function for score calculation
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)
        
        print("IndoBART model loaded successfully!")

    def score(self, srcs, tgts, batch_size=4):
        """
        Score a batch of examples (source -> target direction)
        
        Args:
            srcs (list): List of source texts
            tgts (list): List of target texts
            batch_size (int): Batch size for processing
            
        Returns:
            list: Scores for each source-target pair (higher is better)
        """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    # Prepare input with language tokens
                    encoded_src = self.tokenizer.prepare_input_for_generation(
                        src_list,
                        return_tensors='pt',
                        lang_token=self.lang_token,
                        decoder_lang_token=self.lang_token
                    )
                    
                    # For target, we need to tokenize manually to get labels
                    encoded_tgt_raw = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)
                    tgt_tokens = encoded_tgt_raw['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt_raw['attention_mask'].to(self.device)
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    # Get model output
                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    
                    # Calculate scores similar to BARTScore
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]  # Higher score is better
                    score_list += curr_score_list

            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f'Source: {src_list}')
                print(f'Target: {tgt_list}')
                # Continue with next batch instead of exiting
                continue
                
        return score_list


def evaluate_indonesian_summarization(source_texts, summaries, batch_size=4):
    """
    Evaluate Indonesian summaries using BARTScore approach with IndoBART.
    
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
    
    # Calculate harmonic mean (F-score)
    print("Calculating combined scores...")
    avg_scores = [(s2s + s2t)/2 for s2s, s2t in zip(src_to_sum_scores, sum_to_src_scores)]
    
    return {
        "src_to_sum": src_to_sum_scores,  # Faithfulness: does the summary capture the source
        "sum_to_src": sum_to_src_scores,  # Relevance: is the summary content in the source
        "avg_score": avg_scores           # Combined score
    }


if __name__ == "__main__":
    # Example Indonesian texts and summaries
    indonesian_sources = [
        "Perekonomian Indonesia sedang mengalami pertumbuhan yang signifikan dalam beberapa tahun terakhir. Berbagai sektor industri mulai berkembang pesat, terutama di bidang teknologi dan pariwisata. Investasi asing juga terus meningkat seiring dengan stabilnya kondisi politik dan keamanan. Pemerintah telah melakukan berbagai upaya untuk meningkatkan iklim investasi, seperti menyederhanakan perizinan dan memberikan insentif pajak. Meskipun demikian, masih ada beberapa tantangan yang dihadapi, seperti infrastruktur yang belum merata dan kesenjangan ekonomi antar daerah.",
        
        "Jakarta adalah ibu kota Indonesia yang merupakan pusat pemerintahan dan ekonomi. Kota ini menghadapi berbagai masalah perkotaan seperti kemacetan, banjir, dan polusi udara. Pemerintah telah menerapkan berbagai kebijakan untuk mengatasi masalah tersebut, termasuk pembangunan transportasi massal dan normalisasi sungai. Meskipun demikian, jumlah penduduk yang terus bertambah dan urbanisasi yang tinggi membuat permasalahan ini sulit diatasi sepenuhnya."
    ]

    indonesian_summaries = [
        "Ekonomi Indonesia tumbuh signifikan dengan perkembangan di sektor teknologi dan pariwisata serta peningkatan investasi asing, namun masih menghadapi tantangan infrastruktur dan kesenjangan ekonomi.",
        
        "Jakarta sebagai ibu kota Indonesia menghadapi masalah kemacetan, banjir, dan polusi, yang diatasi pemerintah dengan transportasi massal dan normalisasi sungai, tetapi urbanisasi tinggi mempersulit penyelesaian."
    ]

    print("=" * 60)
    print("INDONESIAN SUMMARIZATION EVALUATION WITH BARTSCORE")
    print("=" * 60)
    
    # Run evaluation
    results = evaluate_indonesian_summarization(
        source_texts=indonesian_sources,
        summaries=indonesian_summaries,
        batch_size=2
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 60)
    
    for i, (src, summary) in enumerate(zip(indonesian_sources, indonesian_summaries)):
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
