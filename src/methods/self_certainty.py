import torch
import torch.nn.functional as F
from collections import defaultdict


def borda_count(paths, borda_p=1.0):
    # Sort candidates by self-certainty score descending (higher certainty = rank 1)
    sorted_candidates = sorted(paths, key=lambda x: x[1], reverse=True)
    
    N = len(sorted_candidates)
    votes = defaultdict(float)
    answer_map = {}

    # Calculate Borda votes
    for rank_idx, item in enumerate(sorted_candidates):
        answer_text, _, final_answer = item

        # Rank r starts at 1
        r = rank_idx + 1

        # v(r) = (N - r + 1)^p, formula (11) in the paper
        vote_weight = (N - r + 1) ** borda_p
        
        votes[final_answer] += vote_weight

        if final_answer not in answer_map:
            answer_map[final_answer] = answer_text
    
    # Select the answer with the highest accumulated votes
    best_answer = max(votes, key=votes.get)
    best_score = votes[best_answer]
    best_answer_text = answer_map[best_answer]

    return best_answer_text, best_score, best_answer


def self_certainty(sample_paths, method_cfg, tokenizer, config):

    method_records = []

    borda_p = method_cfg.get("borda_p", 1.0)

    for i, path in enumerate(sample_paths):
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_logits = path["output_logits"]

        V = output_logits.size(-1)
        V_tensor = torch.tensor(V, dtype=output_logits.dtype, device=output_logits.device)
        log_probs = F.log_softmax(output_logits, dim=-1)
        logprob_sum = torch.sum(log_probs, dim=-1)

        # Formula: -1/V * sum(log_p) - log(V)
        kl = -1/V * logprob_sum - torch.log(V_tensor)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            kl = kl[mask]

        sc_score = (kl.sum() / len(kl)).item()

        method_records.append((answer_text, sc_score, final_answer))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in self_certainty")
    
    path_info = [
        {"answer_text": a, "score": s, "final_answer": f}
        for (a, s, f) in method_records
    ]
    
    if config.aggregate:
        result = borda_count(method_records, borda_p)
    else:
        result = max(method_records, key=lambda x: x[1])
    
    answer_text, score, final_answer = result
    return answer_text, score, final_answer, path_info
    
