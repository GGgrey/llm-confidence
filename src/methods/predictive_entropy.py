import torch
import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def predictive_entropy(sample_paths, normalized_length, tokenizer, config):
    
    method_records = []

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
        
        total_entropy = entropy.sum()
        normalization_factor = len(entropy) if normalized_length else 1.0

        pe = (total_entropy / normalization_factor).item()

        method_records.append((answer_text, pe, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in predictive_entropy")
    
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