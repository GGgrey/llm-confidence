import numpy as np
import torch
from scipy import stats

from src.utils.utils import aggregate_paths_based_on_scores


def calculate_confidence_for_tokens(logits, selected_tokens_idx, selected_tokens_ids, confidence_method="default"):
    
    if confidence_method == "product":
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


def get_attn_vert_scores(matrix, proximity_ignore=4):
    seq_len = matrix.shape[0]
    vert_scores = []
    for i in range(seq_len):
        vert_lines = matrix[i + proximity_ignore:, i]
        vert_score = np.nanmean(vert_lines) if len(vert_lines) > 0 else np.nan
        vert_scores.append(vert_score)
    return np.array(vert_scores)


def compute_key_confidence(answer_ids, prompt_len, attn_weights, output_scores, confidence_method):

    confidence = 0.0

    if attn_weights is None:
        return confidence
    
    num_layers = len(attn_weights)
    num_heads = attn_weights[0].shape[1]
    kurtosis_list = []

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            layer_attn_full = attn_weights[layer_idx][0, head_idx].to(torch.float).cpu().numpy()  # (seq, seq)
            layer_attn = layer_attn_full[prompt_len:, prompt_len:]  # (answer_len, answer_len)
            vert_scores = get_attn_vert_scores(layer_attn, proximity_ignore=4)
            kurt = stats.kurtosis(vert_scores, fisher=True, bias=True, nan_policy="omit")
            kurtosis_list.append((kurt, layer_idx, head_idx))

    kurtosis_list = [entry for entry in kurtosis_list if entry[1] != 0]
    kurtosis_list.sort(reverse=True, key=lambda x: x[0])
    top_heads = kurtosis_list[:3]

    output_len = answer_ids.shape[0]
    token_scores = np.zeros(output_len)
    token_counts = np.zeros(output_len)

    for _, layer_idx, head_idx in top_heads:
        layer_attn_full = attn_weights[layer_idx][0, head_idx].to(torch.float).cpu().numpy()
        layer_attn = layer_attn_full[prompt_len:, prompt_len:]
        for j in range(output_len):
            lower_indices = np.arange(j+1, output_len)
            values = layer_attn[lower_indices, j]
            token_scores[j] += values.sum()
            token_counts[j] += (values != 0).sum()

    token_avgs = np.divide(token_scores, token_counts, out=np.zeros_like(token_scores), where=token_counts!=0)
    token_order = np.argsort(-token_avgs)
    percentile_threshold = 20
    tau = np.percentile(token_avgs, percentile_threshold)

    selected_tokens_idx = [idx for idx in token_order if token_avgs[idx] >= tau]

    if len(selected_tokens_idx) == 0:
        return confidence
    
    selected_tokens_idx = torch.tensor(selected_tokens_idx, dtype=torch.long)
    # selected_tokens_scores = output_scores[selected_tokens_idx, :]
    selected_tokens_ids = answer_ids[selected_tokens_idx]

    confidence = calculate_confidence_for_tokens(output_scores, selected_tokens_idx, selected_tokens_ids, confidence_method)

    return confidence


def key_confidence(sample_paths, method_cfg, model, tokenizer, config):
    
    method_records = []
    confidence_method = method_cfg["confidence"]

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]
        prompt_len = path["prompt_len"]

        if tokenizer.pad_token_id is not None:
            mask = generated_ids != tokenizer.pad_token_id
            generated_ids = generated_ids[mask]

            mask = answer_ids != tokenizer.pad_token_id
            answer_ids = answer_ids[mask]
            output_scores = output_scores[mask]

            prompt_len = len(generated_ids) - len(answer_ids)

        attn_weights = get_attn_weights(generated_ids, model)

        confidence = compute_key_confidence(answer_ids, prompt_len, attn_weights, output_scores, confidence_method)

        method_records.append((answer_text, confidence, final_answer))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in key_confidence")
    
    path_info = [
        {"answer_text": a, "score": s, "final_answer": f}
        for (a, s, f) in method_records
    ]

    if config.aggregate:
        result = aggregate_paths_based_on_scores(method_records)
    else:
        result = max(method_records, key=lambda x: x[1])
    
    answer_text, score, final_answer = result
    return answer_text, score, final_answer, path_info

