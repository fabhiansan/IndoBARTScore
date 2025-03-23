#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IndoBARTScore - Adaptation of BARTScore for Indonesian text evaluation using IndoBART.
Based on the original BARTScore implementation but modified to work with IndoBART.
"""

import torch
import torch.nn as nn
import traceback
import sys
import os
from typing import List
import numpy as np
from transformers import MBartForConditionalGeneration

# Add path to indobenchmark-toolkit
sys.path.append(os.path.join(os.path.dirname(__file__), 'indobenchmark-toolkit/src'))
from indobenchmark.tokenization_indonlg import IndoNLGTokenizer


class IndoBARTScorer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', max_length=1024, checkpoint='indobenchmark/indobart-v2'):
        # Set up model
        self.device = device
        self.max_length = max_length
        print(f"Loading IndoBART model from {checkpoint}...")
        self.tokenizer = IndoNLGTokenizer.from_pretrained(checkpoint, additional_special_tokens=[])
        self.model = MBartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        
        # Language token for Indonesian
        self.lang_token = '[indonesian]'

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)
        print("IndoBART model loaded successfully!")

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            raise ValueError("Path must be provided for loading custom weights")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded model weights from {path}")

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples using IndoBART """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    # Prepare inputs for IndoBART with language tokens
                    encoded_src = self.tokenizer.prepare_input_for_generation(
                        src_list,
                        return_tensors='pt',
                        lang_token=self.lang_token, 
                        decoder_lang_token=self.lang_token
                    )
                    
                    # For target, we need to manually tokenize
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask'].to(self.device)
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    # Get model output with labels for loss calculation
                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    
                    # Calculate log-likelihood scores
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]  # Higher score is better
                    score_list += curr_score_list

            except Exception as e:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        """ Score with multiple references, same as original BARTScore """
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

    def test(self, batch_size=3):
        """ Test IndoBARTScore with sample Indonesian texts """
        src_list = [
            "Indonesia adalah negara kepulauan terbesar di dunia.",
            "Jakarta adalah ibu kota Indonesia.",
            "Bahasa resmi Indonesia adalah Bahasa Indonesia."
        ]

        tgt_list = [
            "Indonesia memiliki banyak pulau.",
            "Jakarta merupakan kota dengan penduduk terpadat di Indonesia.",
            "Bahasa Indonesia adalah bahasa nasional."
        ]

        print("Testing IndoBARTScore with sample texts:")
        for src, tgt in zip(src_list, tgt_list):
            print(f"Source: {src}")
            print(f"Target: {tgt}")
        print("\nScores:")
        scores = self.score(src_list, tgt_list, batch_size)
        for src, tgt, score in zip(src_list, tgt_list, scores):
            print(f"{src} → {tgt}: {score:.4f}")
        
        return scores


if __name__ == "__main__":
    # Example usage
    scorer = IndoBARTScorer()
    scorer.test()
