import numpy as np
import torch
import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def get_attn_weights(generated_ids, model):

    attn_weights = None

    if generated_ids.device != model.device:
        generated_ids = generated_ids.to(model.device)

    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            output_attentions=True,
            return_dict=True
        )

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            attn_weights = outputs.attentions  # tuple: (num_layers, batch, num_heads, seq, seq)

        del outputs
    
    if attn_weights is None:
        return None
    
    return attn_weights


def compute_attention_entropy(attention_weights):

    attn_weights = torch.clamp(attention_weights, min=1e-9)

    log_attn_weights = torch.log(attn_weights)
    entropy_per_token = -torch.sum(attn_weights * log_attn_weights, dim=-1)

    avg_entropy_per_head = torch.mean(entropy_per_token, dim=-1)

    return avg_entropy_per_head


def aggregate_attention_entropy(attention_entropy, method="mean", top_k=5):
    if method == "mean":
        return attention_entropy.mean()
    elif method == "min":
        return attention_entropy.min()
    elif method == "max":
        return attention_entropy.max()
    elif method == "top_k_mean":
        if top_k >= len(attention_entropy):
            return attention_entropy.mean()
        sorted_entropies, _ = torch.sort(attention_entropy, descending=False)
        return sorted_entropies[:top_k].mean()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def attention_entropy(sample_paths, method_cfg, model, tokenizer, config):
    
    method_records = []

    scoring_mode = method_cfg.get("scoring_mode", "mean")
    top_k = method_cfg.get("top_k", 5)

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        answer_ids = path["answer_ids"]

        # Filter padding tokens
        if tokenizer.pad_token_id is not None:
            mask = generated_ids != tokenizer.pad_token_id
            generated_ids = generated_ids[mask]

            mask = answer_ids != tokenizer.pad_token_id
            answer_ids = answer_ids[mask]

        attn_weights = get_attn_weights(generated_ids, model)
        if attn_weights is None:
             raise RuntimeError("Get model attention weights failed")

        prompt_len = len(generated_ids) - len(answer_ids)

        # Use last layer
        last_layer_attn_weights = attn_weights[-1].squeeze(0)

        output_attn_weights = last_layer_attn_weights[:, prompt_len:, :]

        attn_entropy = compute_attention_entropy(output_attn_weights)

        score = aggregate_attention_entropy(
            attn_entropy, 
            method=scoring_mode, 
            top_k=top_k
        )

        method_records.append((answer_text, score, final_answer))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in attention_entropy")
    
    path_info = [
        {"answer_text": a, "score": s, "final_answer": f}
        for (a, s, f) in method_records
    ]

    if config.aggregate:
        result = aggregate_paths_based_on_scores_using_min(method_records)
    else:
        result = min(method_records, key=lambda x: x[1])

    answer_text, score, final_answer = result
    return answer_text, score, final_answer, path_info