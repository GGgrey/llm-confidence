import torch
import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def group_entropy(sample_paths, method_cfg, tokenizer, config):
    
    method_records = []
    k = method_cfg["k"]

    all_entropy = []

    for path in sample_paths:
        answer_ids = path["answer_ids"]
        output_scores = path["output_scores"]

        probs = F.softmax(output_scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            entropy = entropy[mask]

        if entropy.numel() > 0:
            all_entropy.append(entropy)

    if len(all_entropy) == 0:
        raise RuntimeError("No valid tokens to compute entropy")

    all_entropy = torch.cat(all_entropy, dim=0)
    mean_entropy = all_entropy.mean()
    std_entropy = all_entropy.std(unbiased=True) if all_entropy.numel() > 1 else torch.tensor(0.0, device=all_entropy.device)
    high_entropy_threshold = mean_entropy + k * std_entropy

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

        high_entropy_mask = entropy > high_entropy_threshold
        selected_entropy = entropy[high_entropy_mask]

        if len(selected_entropy) > 0:
            confidence = selected_entropy.mean().item()
        else:
            confidence = 0.0
        
        method_records.append((answer_text, confidence, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in group_entropy")
    
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

    