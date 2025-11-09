import torch
import torch.nn.functional as F

from src.utils import aggregate_paths_based_on_scores_using_min


def group_entropy(sample_paths, tokenizer, config, k=3):
    
    method_records = []
    entropy_list = []

    for path in sample_paths:
        answer_ids = path["answer_ids"]
        output_scores = path["output_scores"]

        probs = F.softmax(output_scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            entropy = entropy[mask]

        total_entropy = entropy.sum()
        normalized_entropy = (total_entropy / len(entropy)).item()
        entropy_list.append(normalized_entropy)

    mean_entropy = float(torch.tensor(entropy_list).mean())
    std_entropy = float(torch.tensor(entropy_list).std(unbiased=True))
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
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores_using_min(method_records)
    else:
        return (min(method_records, key=lambda x: x[1]))

    