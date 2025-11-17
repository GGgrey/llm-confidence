from typing import List, Dict, Tuple

import numpy as np
import torch

from src.utils import aggregate_paths_based_on_scores


def get_attn_weights(generated_ids, model):
    """提取注意力权重。"""
    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            output_attentions=True,
            return_dict=True
        )
        attn_weights = getattr(outputs, 'attentions', None)
    if attn_weights is None:
        return None
    return attn_weights  # tuple: (num_layers, batch, num_heads, seq, seq)


def attention_enhanced(sample_paths, method_cfg, model, tokenizer, config):

    method_records = []

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        answer_ids = path["answer_ids"]
        output_scores = path["output_scores"]
        prompt_len = path["prompt_len"]
        prompt = path["prompt"]

        attn_weights = get_attn_weights(generated_ids, model)
        if attn_weights is None:
            raise RuntimeError("No attention weights from model.")
        num_layers = len(attn_weights)
        num_heads = attn_weights[0].shape[1]
        seq_len = len(generated_ids)

        all_attn_maps = {
            (layer_idx, head_idx): attn_weights[layer_idx][0, head_idx].float().cpu().numpy()
            for layer_idx in range(num_layers)
            for head_idx in range(num_heads)
        }

        prompt_tokens = tokenizer.tokenize(prompt)
        key_token_indices = [i for i in range(prompt_len) if prompt_tokens[i].isdigit()]
        anomaly_token_indices = [i for i in range(prompt_len) if 'not' in prompt_tokens[i].lower()]

        