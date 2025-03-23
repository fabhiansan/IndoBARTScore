#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IndoBARTScore - Adaptation of BARTScore for Indonesian text evaluation using IndoBART.
Based on the structure in BARTScore/SUM/bart_score.py
"""

import torch
import torch.nn as nn
import traceback
import sys
import os
from transformers import MBartForConditionalGeneration

# Add path to indobenchmark-toolkit
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'indobenchmark-toolkit/src'))
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

    def load(self, path):
        """ Load custom model weights """
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
