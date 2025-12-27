import torch
import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def compute_eu(logits, k=2):
    # Extract top-k values
    topk_values, _ = torch.topk(logits, k, dim=-1)

    # Apply ReLU
    alpha = torch.clamp(topk_values, min=0.0)

    # Sum of evidence
    sum_alpha = alpha.sum(dim=-1)
    
    # Calculate EU
    # Formula: top_k / (np.sum(...) + top_k)
    eu_score = k / (sum_alpha + k)

    return eu_score


def compute_au(logits, k=2):
    # Extract top-k values
    topk_values, _ = torch.topk(logits, k, dim=-1)
    
    # Apply ReLU
    alpha = torch.clamp(topk_values, min=0.0)
    
    # Sum of alpha (alpha_0)
    alpha_0 = alpha.sum(dim=-1, keepdim=True)

    # Compute digamma functions
    psi_alpha_plus_1 = torch.digamma(alpha + 1.0)
    psi_alpha_0_plus_1 = torch.digamma(alpha_0 + 1.0)

    # Calculate AU
    # Formula: - sum((alpha / alpha_0) * (psi(alpha+1) - psi(alpha_0+1)))
    # Add epsilon for numerical stability to avoid div by zero if sum is 0
    epsilon = 1e-10
    term = (alpha / (alpha_0 + epsilon)) * (psi_alpha_plus_1 - psi_alpha_0_plus_1)

    # Sum over k dimensions and negate
    au_score = -term.sum(dim=-1)

    return au_score


def logtoku(sample_paths, method_cfg, tokenizer, config):

    method_records = []
    
    k_logits = method_cfg.get("k_logits", 5)  # The hyperparameter 'K' in Eq. (3)
    k_worst_tokens = method_cfg.get("k_worst_tokens", 25)  # Define the size of set T_K in Eq. (7)

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]  # [seq_len, vocab_size]

        # Compute EU and AU separately
        eu_scores = compute_eu(output_scores, k=k_logits)
        au_scores = compute_au(output_scores, k=k_logits)

        # Combine EU and AU
        token_uncertainty = eu_scores * au_scores

        # Filtering padding token
        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            token_uncertainty = token_uncertainty[mask]

        if len(token_uncertainty) > 0:
            k_actual = min(k_worst_tokens, len(token_uncertainty))
            top_uncertainties, _ = torch.topk(token_uncertainty, k_actual)
            final_score = top_uncertainties.mean().item()
        else:
            final_score = float('inf') 

        method_records.append((answer_text, final_score, final_answer))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in LogTokU")
    
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