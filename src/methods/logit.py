import torch
import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores 


def logit_scoring(sample_paths, method_cfg, tokenizer, config):

    method_records = []
    scoring_mode = method_cfg.get("scoring_mode", "mean")
    top_k = method_cfg.get("top_k", 20)

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        # Filter padding tokens
        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            output_scores = output_scores[mask]
            answer_ids = answer_ids[mask]

        chosen_token_logits = output_scores.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1)

        topk_token_logits, _ = torch.topk(output_scores, k=top_k, dim=-1)
        topk_token_logits = torch.clamp(topk_token_logits, min=0.0)

        score = 0.0
        if scoring_mode == "mean":
            score = chosen_token_logits.mean().item()
        elif scoring_mode == "min":
            score = chosen_token_logits.min().item()
        elif scoring_mode == "max":
            score = chosen_token_logits.max().item()
        elif scoring_mode == "gap":
            max_logits = topk_token_logits[:, 0]
            rest_logits = topk_token_logits[:, 1:top_k]
            token_gap_scores = max_logits - rest_logits.mean(dim=-1)
            score = token_gap_scores.mean().item()
        else:
            raise ValueError(f"Unknown scoring mode: {scoring_mode}")

        method_records.append((answer_text, score, final_answer))
        
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in logit_scoring")
    
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