import numpy as np
import torch
import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores


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


def energy(sample_paths, method_cfg, tokenizer, config):

    method_records = []
    
    temperature = method_cfg.get("temperature", 1.0)
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

        # Calculate energy
        energy = temperature * torch.logsumexp(output_scores / temperature, dim=-1)
        energy = energy.detach().cpu().numpy()
        seq_len = len(energy)
        trace_score = 0.0

        if mode == "mean":
            total_energy = energy.sum()
            trace_score = np.mean(energy) if seq_len > 0 else 0.0
        elif mode == "worst":
            if seq_len > 0:
                worst_ratio = method_cfg.get("worst_ratio", 0.2)
                num_worst = int(seq_len * worst_ratio)
                worst_energy = np.partition(energy, num_worst - 1)[:num_worst]
                trace_score = np.mean(worst_energy) if len(worst_energy) > 0 else 0.0
            else:
                trace_score = 0.0
        elif mode == "bottom_group":
            trace_score = calculate_bottom_window_confidence(
                energy, window_size=window_size, bottom_percent=bottom_percent
            )
        elif mode == "lowest_group":
            trace_score = calculate_bottom_window_confidence(
                energy, window_size=window_size, bottom_percent=bottom_percent
            )
        elif mode == "tail":
            tail_count = method_cfg.get("tail_count", 50)
            tail_energy = energy[-tail_count:] if seq_len > tail_count else energy
            trace_score = np.mean(tail_energy) if len(tail_energy) > 0 else 0.0
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        method_records.append((answer_text, trace_score, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in energy")
    
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