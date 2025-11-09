import numpy as np
import torch

from src.utils import aggregate_paths_based_on_scores


def calculate_confidence_for_tokens(logits, selected_tokens_idx, selected_tokens_ids, confidence_method="default"):
    
    if confidence_method == "default":
        confidence = 1.0
    elif confidence_method == "sum":
        confidence = 0.0
    elif confidence_method == "entropy":
        confidence = 0.0
    else:
        raise NotImplementedError(f"Unsupported confidence calculation mode: {confidence_method}")
    valid_tokens = 0

    if len(selected_tokens_ids) != len(selected_tokens_idx):
        return 0.0

    for token_idx, token_id in zip(selected_tokens_idx.tolist(), selected_tokens_ids.tolist()):
        if token_idx >= len(logits):
            break

        token_logits = logits[token_idx]

        probs = torch.softmax(token_logits, dim=-1)
        token_prob = probs[token_id]

        if confidence_method == "default":
            confidence *= token_prob.item()
        elif confidence_method == "sum":
            confidence += token_prob.item()
        elif confidence_method == "entropy":
            probs = torch.clamp(probs, min=1e-12)
            entropy = - (probs * torch.log(probs)).sum(dim=-1)
            vocab_size = probs.size(-1)
            max_entropy = torch.log(torch.tensor(vocab_size, dtype=probs.dtype, device=probs.device))
            norm_entropy = entropy / max_entropy
            confidence += (1.0 - norm_entropy.item())
        else:
            raise NotImplementedError(f"Unsupported confidence calculation mode: {confidence_method}")
        
        valid_tokens += 1

    return confidence / valid_tokens if valid_tokens > 0 else 0.0


def get_attn_weights(generated_ids, model):

    attn_weights = None

    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            output_attentions=True,
            return_dict=True
        )

        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attn_weights = outputs.attentions  # tuple: (num_layers, batch, num_heads, seq, seq)
    
    if attn_weights is None:
        return None
    
    return attn_weights


def compute_attn_eigenvalue(attn_weights, generated_ids, answer_ids, prompt_len, output_scores, scoring_mode, confidence_method, tokenizer, use_len=True, top_k=20):

    scores = []

    for layer_num in range(1, len(attn_weights)):
        eig_score = 0.0
        conf_score = 0.0
        for head_num in range(len(attn_weights[layer_num])):
            sigma = attn_weights[layer_num][head_num]

            if use_len:
                sigma = sigma[prompt_len: len(generated_ids), prompt_len: len(generated_ids)]

            diag_vals = torch.log(torch.diagonal(sigma, 0))

            eig_score += diag_vals.mean()

            actual_k = min(top_k, diag_vals.shape[0])
            _, selected_tokens_idx = torch.topk(-diag_vals, actual_k)
            selected_tokens_ids = answer_ids[selected_tokens_idx]
            conf_score += calculate_confidence_for_tokens(output_scores, selected_tokens_idx, selected_tokens_ids, confidence_method)
        
        if scoring_mode == "sum":
            scores.append(-eig_score.item())
        elif scoring_mode == "top_k":
            scores.append(conf_score)
    
    return np.mean(scores).item()


def attention_eigenvalue(sample_paths, method_cfg, model, tokenizer, config):
    
    method_records = []
    scoring_mode = method_cfg["scoring_mode"]
    confidence_method = method_cfg["confidence"]

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]
        prompt_len = path["prompt_len"]

        attn_weights = get_attn_weights(generated_ids, model)
        attn_weights = [x[0].to(torch.float32).detach().cpu() for x in attn_weights]

        confidence = compute_attn_eigenvalue(attn_weights, generated_ids, answer_ids, prompt_len, output_scores, scoring_mode, confidence_method, tokenizer)

        method_records.append((answer_text, confidence, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")

    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return (max(method_records, key=lambda x: x[1]))