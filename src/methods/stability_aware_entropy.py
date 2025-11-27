import torch
import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def stability_aware_entropy(sample_paths, method_cfg, tokenizer, config, window_size=10, alpha=0.5, beta=0.2):

    method_records = []

    window_size = method_cfg.get("window_size", window_size)
    alpha = method_cfg.get("alpha", alpha)
    beta = method_cfg.get("beta", beta)

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        probs = F.softmax(output_scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            entropy = entropy[mask]

        length = len(entropy)

        global_norm_entropy = (entropy.sum() / length).item()

        if length < window_size:
            peak_window_entropy = global_norm_entropy
        else:
            windows = entropy.unfold(dimension=0, size=window_size, step=1)
            window_means = windows.mean(dim=-1)
            peak_window_entropy = window_means.max().item()

        if length > 1:
            volatility = entropy.std(unbiased=True).item()
        else:
            volatility = 0.0

        trajectory_score = global_norm_entropy + (alpha * peak_window_entropy) + (beta * volatility)

        method_records.append((answer_text, trajectory_score, final_answer))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error in entropy trajectory calculation")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores_using_min(method_records)
    else:
        return min(method_records, key=lambda x: x[1])
