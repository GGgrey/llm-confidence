import torch
import torch.nn.functional as F

from src.utils import aggregate_paths_based_on_scores


def gibbs_entropy_lin(probs, vocab_size, alpha=1.0):
    """Linearly normalized Gibbs entropy-based confidence"""
    H_g = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    max_H_g = torch.log(torch.tensor(vocab_size, dtype=probs.dtype, device=probs.device))
    return 1 - H_g / max_H_g


def gibbs_entropy_exp(probs, vocab_size, alpha=1.0):
    """Exponentially normalized Gibbs entropy-based confidence"""
    sum_p_log_p = torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    return (vocab_size * torch.exp(sum_p_log_p) - 1) / (vocab_size - 1)


def tsallis_entropy_lin(probs, vocab_size, alpha):
    """Linearly normalized Tsallis entropy-based confidence"""
    V = torch.tensor(vocab_size, dtype=probs.dtype, device=probs.device)

    if abs(alpha - 1.0) < 1e-8:
        H_g = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        max_H_g = torch.log(V)
        return 1 - H_g / max_H_g

    sum_p_alpha = torch.sum(probs ** alpha, dim=-1)
    return (V ** (1 - alpha) - sum_p_alpha) / (V ** (1 - alpha) - 1)


def tsallis_entropy_exp(probs, vocab_size, alpha):
    """Exponentially normalized Tsallis entropy-based confidence"""
    V = torch.tensor(vocab_size, dtype=probs.dtype, device=probs.device)

    if abs(alpha - 1.0) < 1e-8:
        sum_p_log_p = torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        return (vocab_size * torch.exp(sum_p_log_p) - 1) / (vocab_size - 1)

    sum_p_alpha = torch.sum(probs ** alpha, dim=-1)
    num = torch.exp((V ** (1 - alpha) - sum_p_alpha) / (1 - alpha)) - 1
    den = torch.exp((V ** (1 - alpha) - 1) / (1 - alpha)) - 1
    return num / (den + 1e-12)


def tsallis_entropy_exp_stable(probs, vocab_size, alpha):
    V = torch.tensor(vocab_size, dtype=probs.dtype, device=probs.device)

    if abs(alpha - 1.0) < 1e-8:
        sum_p_log_p = torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        return (vocab_size * torch.exp(sum_p_log_p) - 1) / (vocab_size - 1)

    sum_p_alpha = torch.sum(probs ** alpha, dim=-1)
    A = (V ** (1 - alpha) - sum_p_alpha) / (1 - alpha)
    B = (V ** (1 - alpha) - 1) / (1 - alpha)

    threshold = 50.0

    if A.ndim == 0:
        if (A > threshold) and (B > threshold):
            return torch.exp(A - B)
        else:
            return torch.expm1(A) / (torch.expm1(B) + 1e-12)
        
    large_mask = (A > threshold) & (B > threshold)
    F = torch.empty_like(A)

    F[~large_mask] = torch.expm1(A[~large_mask]) / (torch.expm1(B) + 1e-12)
    F[large_mask] = torch.exp(A[large_mask] - B)

    return F


def renyi_entropy_lin(probs, vocab_size, alpha):
    """Linearly normalized Rényi entropy-based confidence"""
    V = torch.tensor(vocab_size, dtype=probs.dtype, device=probs.device)

    if abs(alpha - 1.0) < 1e-8:
        H_g = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        max_H_g = torch.log(V)
        return 1 - H_g / max_H_g

    sum_p_alpha = torch.sum(probs ** alpha, dim=-1)
    log_base_V = torch.log(sum_p_alpha + 1e-12) / torch.log(V)
    return 1 + log_base_V / (alpha - 1)


def renyi_entropy_exp(probs, vocab_size, alpha):
    """Exponentially normalized Rényi entropy-based confidence"""
    V = torch.tensor(vocab_size, dtype=probs.dtype, device=probs.device)
    sum_p_alpha = torch.sum(probs ** alpha, dim=-1)
    return (V * (sum_p_alpha ** (1 / (alpha - 1))) - 1) / (V - 1)


def xentropy(sample_paths, method_cfg, tokenizer, config):
    
    method_records = []
    scoring_mode = method_cfg["scoring_mode"]
    confidence_method = method_cfg["confidence"]
    alpha = method_cfg.get("alpha", 1.0)

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]
        answer_ids = path["answer_ids"]

        probs = F.softmax(output_scores, dim=-1)
        vocab_size = probs.size(-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            probs = probs[mask]

        if confidence_method == "gibbs_entropy_lin":
            entropy = gibbs_entropy_lin(probs, vocab_size, alpha)
        elif confidence_method == "gibbs_entropy_exp":
            entropy = gibbs_entropy_exp(probs, vocab_size, alpha)
        elif confidence_method == "tsallis_entropy_lin":
            entropy = tsallis_entropy_lin(probs, vocab_size, alpha)
        elif confidence_method == "tsallis_entropy_exp":
            entropy = tsallis_entropy_exp_stable(probs, vocab_size, alpha)
        elif confidence_method == "renyi_entropy_lin":
            entropy = renyi_entropy_lin(probs, vocab_size, alpha)
        elif confidence_method == "renyi_entropy_exp":
            entropy = renyi_entropy_exp(probs, vocab_size, alpha)
        else:
            raise ValueError(f"Unknown confidence method: {confidence_method}")
        
        if scoring_mode == "min":
            confidence = torch.min(entropy, dim=-1)[0]
        elif scoring_mode == "max":
            confidence = torch.max(entropy, dim=-1)[0]
        elif scoring_mode == "mean":
            confidence = torch.mean(entropy, dim=-1)
        elif scoring_mode == "product":
            confidence = torch.prod(entropy, dim=-1)
        else:
            raise ValueError(f"Unknown scoring mode: {scoring_mode}")
        
        method_records.append((answer_text, confidence.item(), final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return (max(method_records, key=lambda x: x[1]))

