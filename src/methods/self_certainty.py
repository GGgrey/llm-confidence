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
        output_scores = path["output_scores"]

        # Get probabilities distribution
        probs = F.softmax(output_scores, dim=-1)

        # Get vocabulary size (V)
        vocab_size = output_scores.size(-1)

        # Calculate self-certainty
        log_term = torch.log(vocab_size * probs + 1e-9)
        kl = - (1.0 / vocab_size) * torch.sum(log_term, dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            kl = kl[mask]

        sc_score = (kl.sum() / len(kl)).item()

        method_records.append((answer_text, sc_score, final_answer))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return borda_count(method_records, borda_p)
    else:
        return (max(method_records, key=lambda x: x[1]))
    
