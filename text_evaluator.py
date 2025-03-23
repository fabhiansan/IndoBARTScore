#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate text summarization using BARTScore with IndoBART.
"""

import os
import sys
import argparse
import torch
import numpy as np
import torch.nn as nn

# Import BARTScore
sys.path.append(os.path.join(os.path.dirname(__file__), 'BARTScore'))
from BARTScore.bart_score import BARTScorer

# Import IndoBART tokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'indobenchmark-toolkit'))
from indobenchmark import IndoNLGTokenizer
from transformers import MBartForConditionalGeneration


class IndoBARTScorer:
    """BARTScore implementation that uses IndoBART model."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', max_length=1024, checkpoint='indobenchmark/indobart-v2'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = IndoNLGTokenizer.from_pretrained(checkpoint, additional_special_tokens=[])
        self.model = MBartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        
        # Language token for Indonesian
        self.lang_token = '[indonesian]'

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples (similar to BARTScore) """
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
                        decoder_lang_token=self.lang_token,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True
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
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                # Continue with next batch instead of exiting
                continue
                
        return score_list

    def multi_ref_score(self, srcs, tgts, agg="mean", batch_size=4):
        """ Score with multiple references, similar to BARTScore implementation """
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)


class TextEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the text evaluator with both BARTScore and IndoBARTScorer."""
        self.device = device
        print(f"Using device: {device}")
        
        # Initialize BARTScore with English BART
        print("Loading standard BARTScore model...")
        self.bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        
        # Initialize IndoBARTScorer
        print("Loading IndoBART for BARTScore...")
        self.indobart_scorer = IndoBARTScorer(device=device)
        
        print("Models loaded successfully!")

    def evaluate_english_summarization(self, source_texts, summaries, batch_size=4):
        """
        Evaluate English summaries using BARTScore.
        
        Args:
            source_texts (list): List of source texts
            summaries (list): List of summaries to evaluate
            batch_size (int): Batch size for processing
            
        Returns:
            dict: Dictionary with evaluation results
        """
        # Score from source to summary (tests faithfulness)
        src_to_sum_scores = self.bart_scorer.score(source_texts, summaries, batch_size)
        
        # Score from summary to source (tests relevance)
        sum_to_src_scores = self.bart_scorer.score(summaries, source_texts, batch_size)
        
        # Average the two directions for a more balanced score
        avg_scores = [(s2s + s2t)/2 for s2s, s2t in zip(src_to_sum_scores, sum_to_src_scores)]
        
        return {
            "src_to_sum": src_to_sum_scores,  # Faithfulness: does the summary capture the source
            "sum_to_src": sum_to_src_scores,  # Relevance: is the summary content in the source
            "avg_score": avg_scores           # Combined score
        }
    
    def evaluate_indonesian_summarization(self, source_texts, summaries, batch_size=4):
        """
        Evaluate Indonesian summaries using IndoBARTScorer.
        
        Args:
            source_texts (list): List of source texts
            summaries (list): List of summaries to evaluate
            batch_size (int): Batch size for processing
            
        Returns:
            dict: Dictionary with evaluation results
        """
        # Score from source to summary (tests faithfulness)
        src_to_sum_scores = self.indobart_scorer.score(source_texts, summaries, batch_size)
        
        # Score from summary to source (tests relevance)
        sum_to_src_scores = self.indobart_scorer.score(summaries, source_texts, batch_size)
        
        # Average the two directions for a more balanced score
        avg_scores = [(s2s + s2t)/2 for s2s, s2t in zip(src_to_sum_scores, sum_to_src_scores)]
        
        return {
            "src_to_sum": src_to_sum_scores,  # Faithfulness: does the summary capture the source
            "sum_to_src": sum_to_src_scores,  # Relevance: is the summary content in the source
            "avg_score": avg_scores           # Combined score
        }


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Evaluate text summarization using BARTScore")
    parser.add_argument("--source", "-s", required=True, type=str, help="Source text or file with texts")
    parser.add_argument("--summary", "-r", required=True, type=str, help="Summary text or file with summaries")
    parser.add_argument("--indonesian", "-i", action="store_true", help="Set if texts are in Indonesian")
    parser.add_argument("--batch", "-b", action="store_true", help="Set if inputs are files with one text per line")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    args = parser.parse_args()
    
    # Initialize the evaluator
    evaluator = TextEvaluator()
    
    # Process input
    if args.batch:
        # Read from files
        with open(args.source, 'r', encoding='utf-8') as f:
            source_texts = [line.strip() for line in f if line.strip()]
            
        with open(args.summary, 'r', encoding='utf-8') as f:
            summaries = [line.strip() for line in f if line.strip()]
            
        if len(source_texts) != len(summaries):
            print("Error: Number of source texts and summaries must match.")
            sys.exit(1)
    else:
        # Use direct input
        source_texts = [args.source]
        summaries = [args.summary]
    
    # Run evaluation
    if args.indonesian:
        results = evaluator.evaluate_indonesian_summarization(
            source_texts=source_texts,
            summaries=summaries,
            batch_size=args.batch_size
        )
        model_name = "IndoBART-BARTScore"
    else:
        results = evaluator.evaluate_english_summarization(
            source_texts=source_texts,
            summaries=summaries,
            batch_size=args.batch_size
        )
        model_name = "BARTScore"
    
    # Print results
    print(f"\nSummarization Evaluation Results ({model_name}):")
    print("-" * 60)
    
    for i, (src, sum_text) in enumerate(zip(source_texts, summaries)):
        print(f"Example {i+1}:")
        # Truncate long texts for display
        disp_src = (src[:100] + '...') if len(src) > 100 else src
        disp_sum = (sum_text[:100] + '...') if len(sum_text) > 100 else sum_text
        
        print(f"Source: {disp_src}")
        print(f"Summary: {disp_sum}")
        print(f"Source → Summary (Faithfulness): {results['src_to_sum'][i]:.4f}")
        print(f"Summary → Source (Relevance): {results['sum_to_src'][i]:.4f}")
        print(f"Combined Score: {results['avg_score'][i]:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
