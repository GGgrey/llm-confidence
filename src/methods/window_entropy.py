import torch
import torch.nn.functional as F

from src.utils import aggregate_paths_based_on_scores_using_min


def window_entropy(sample_paths, tokenizer, config, normalized_length=True, top_k=20, window_size=10):
    
    method_records = []

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        if top_k is None:
            probs = F.softmax(output_scores, dim=-1)
        else:
            top_values = torch.topk(output_scores, k=top_k, dim=-1).values
            probs = F.softmax(top_values, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            entropy = entropy[mask]

        if len(entropy) < window_size:
            total_entropy = entropy.sum()
            normalization_factor = len(entropy) if normalized_length else 1.0
            return (total_entropy / normalization_factor).item()
        
        windows = entropy.unfold(dimension=0, size=window_size, step=1)
        window_means = windows.mean(dim=-1)

        max_window_entropy = window_means.max().item()

        method_records.append((answer_text, max_window_entropy, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")

    if config.aggregate:
        return aggregate_paths_based_on_scores_using_min(method_records)
    else:
        return (min(method_records, key=lambda x: x[1]))
