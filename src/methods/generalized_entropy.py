import torch
import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def generalized_entropy(sample_paths, method_cfg, tokenizer, config):

    method_records = []

    gamma = float(method_cfg.get("gamma", 0.5))
    top_m = method_cfg.get("top_m", None)

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        probs = F.softmax(output_scores, dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            probs = probs[mask]

        if top_m is not None and top_m < probs.size(-1):
            # We only care about the values of the top probabilities for the sum
            probs, _ = torch.topk(probs, top_m, dim=-1)

        # Calculate generalized entropy per token
        term = probs.pow(gamma) * (1.0 - probs).pow(gamma)
        gen_scores = term.sum(dim=-1)

        total_gen_score = gen_scores.sum()
        score = (total_gen_score / len(gen_scores)).item()

        method_records.append((answer_text, score, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in generalized_entropy")
    
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
