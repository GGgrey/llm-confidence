import torch
import torch.nn.functional as F
import numpy as np

from src.utils.utils import aggregate_paths_based_on_scores


def compute_token_confidence(output_scores, k=5):
    # Calculate probabilities
    probs = F.softmax(output_scores, dim=-1)

    # Get top-k probabilities, shape: [seq_len, k]
    topk_probs, _ = torch.topk(probs, k, dim=-1)

    # Calculate log probabilities
    log_topk_probs = torch.log(topk_probs + 1e-9)

    # Calculate negative average log-probability, shape: [seq_len]
    token_confidences = -torch.mean(log_topk_probs, dim=-1)
    
    return token_confidences


def calculate_bottom_window_confidence(token_confidence, window_size=25, bottom_percent=0.1):
    try:
        if len(token_confidence) == 0:
            return 0.0
        if len(token_confidence) < window_size:
            return np.mean(token_confidence)
        
        window_means = []
        window_sum = sum(token_confidence[:window_size])
        window_means.append(window_sum / window_size)

        for i in range(1, len(token_confidence) - window_size + 1):
            window_sum = window_sum - token_confidence[i-1] + token_confidence[i + window_size - 1]
            window_means.append(window_sum / window_size)
        
        if not window_means:
            return 0.0
        
        if bottom_percent == -1:  # Lowest group
            return min(window_means)
        
        num_bottom = max(1, int(len(window_means) * bottom_percent))
        if num_bottom == 1:
            return min(window_means)
        else:
            bottom_means = np.partition(window_means, num_bottom-1)[:num_bottom]
            return np.mean(bottom_means)
    except Exception:
        return 0.0


def deepconf(sample_paths, method_cfg, tokenizer, config):

    method_records = []

    mode = method_cfg.get("mode", "")
    top_k = method_cfg.get("top_k", 20)
    window_size = method_cfg.get("window_size", 25)
    bottom_percent = method_cfg.get("bottom_percent", -1)

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        # Filter padding tokens
        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            output_scores = output_scores[mask]

        # Calculate token confidence
        # High confidence corresponds to peaked distributions and greater model certainty
        token_confidence = compute_token_confidence(output_scores, k=top_k)
        token_confidence = token_confidence.detach().cpu().numpy()
        trace_score = 0.0

        if mode == "tail":
            tail_count = method_cfg.get("tail_count", 50)
            tail_confidence = token_confidence[-tail_count:] if len(token_confidence) > tail_count else token_confidence
            trace_score = np.mean(tail_confidence) if len(tail_confidence) > 0 else 0.0
        elif mode == "bottom_group":
            trace_score = calculate_bottom_window_confidence(
                token_confidence, window_size=window_size, bottom_percent=bottom_percent
            )
        elif mode == "lowest_group":
            trace_score = calculate_bottom_window_confidence(
                token_confidence, window_size=window_size, bottom_percent=-1
            )
        elif mode == "mean":
            trace_score = np.mean(token_confidence)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        method_records.append((answer_text, trace_score, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in deepconf")
    
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

