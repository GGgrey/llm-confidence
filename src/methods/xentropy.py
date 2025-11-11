import torch
import torch.nn.functional as F

from src.utils import aggregate_paths_based_on_scores


def neg_gibbs_entropy(log_p):
    p = torch.exp(log_p)
    return torch.sum(p * log_p, dim=-1)


def neg_entropy_alpha(log_p, t):
    return torch.sum(torch.exp(log_p * t), dim=-1)


def neg_gibbs_entropy_alpha(log_p, t):
    p_t = torch.exp(log_p * t)
    return torch.sum(p_t * log_p, dim=-1)


def max_prob_confidence(log_p, V, t):
    if t == 1.0:
        p_max = torch.exp(torch.max(log_p, dim=-1)[0])
        return (p_max * V - 1) / (V - 1)
    else:
        p_max_t = torch.exp(torch.max(log_p, dim=-1)[0] * t)
        return (p_max_t * (V ** t) - 1) / ((V ** t) - 1)


def gibbs_entropy_lin(log_p, V, t):
    if t == 1.0:
        return 1.0 + neg_gibbs_entropy(log_p) / torch.log(torch.tensor(V, dtype=log_p.dtype))
    else:
        return 1.0 + neg_gibbs_entropy_alpha(log_p, t) / torch.log(torch.tensor(V, dtype=log_p.dtype)) / (V ** (1 - t))
    

def gibbs_entropy_exp(log_p, V, t):
    if t == 1.0:
        Hg = neg_gibbs_entropy(log_p)
        return (torch.exp(Hg) * V - 1) / (V - 1)
    else:
        exp_neg_max_ent = V ** (-t * (V ** (1 - t)))
        return (torch.exp(neg_gibbs_entropy_alpha(log_p, t) * t) - exp_neg_max_ent) / (1 - exp_neg_max_ent)
    

def tsallis_entropy_lin(log_p, V, t):
    if t == 1.0:
        return 1.0 + neg_gibbs_entropy(log_p) / torch.log(torch.tensor(V, dtype=log_p.dtype))
    else:
        return 1.0 + (1.0 - neg_entropy_alpha(log_p, t)) / ((V ** (1 - t)) - 1)
    

def tsallis_entropy_exp(log_p, V, t):
    if t == 1.0:
        Hg = neg_gibbs_entropy(log_p)
        return (torch.exp(Hg) * V - 1) / (V - 1)
    else:
        exp_neg_max_ent = torch.exp((1.0 - (V ** (1 - t))) / (1.0 - t))
        num = torch.exp((1.0 - neg_entropy_alpha(log_p, t)) / (1.0 - t)) - exp_neg_max_ent
        denom = 1.0 - exp_neg_max_ent
        return num / denom
    

def renyi_entropy_lin(log_p, V, t):
    if t == 1.0:
        return 1.0 + neg_gibbs_entropy(log_p) / torch.log(torch.tensor(V, dtype=log_p.dtype))
    else:
        return 1.0 + torch.log2(neg_entropy_alpha(log_p, t)) / (t - 1) / torch.log(torch.tensor(V, dtype=log_p.dtype), 2)
    

def renyi_entropy_exp(log_p, V, t):
    if t == 1.0:
        Hg = neg_gibbs_entropy(log_p)
        return (torch.exp(Hg) * V - 1) / (V - 1)
    else:
        return ((neg_entropy_alpha(log_p, t)) ** (1 / (t - 1)) * V - 1) / (V - 1)


def xentropy(sample_paths, method_cfg, config):
    
    method_records = []
    scoring_mode = method_cfg["scoring_mode"]
    confidence_method = method_cfg["confidence"]
    alpha = method_cfg.get("alpha", 1.0)

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        log_probs = F.log_softmax(output_scores, dim=-1)
        vocab_size = log_probs.size(-1)

        if confidence_method == "gibbs_entropy":
            if scoring_mode == "linear_normalization":
                confidence = gibbs_entropy_lin(log_probs, vocab_size, alpha)
            else:
                confidence = gibbs_entropy_exp(log_probs, vocab_size, alpha)
        
        elif confidence_method == "tsallis_entropy":
            if scoring_mode == "linear_normalization":
                confidence = tsallis_entropy_lin(log_probs, vocab_size, alpha)
            else:
                confidence = tsallis_entropy_exp(log_probs, vocab_size, alpha)

        elif confidence_method == "renyi_entropy":
            if scoring_mode == "linear_normalization":
                confidence = renyi_entropy_lin(log_probs, vocab_size, alpha)
            else:
                confidence = renyi_entropy_exp(log_probs, vocab_size, alpha)

        else:
            raise ValueError(f"Unknown confidence method: {confidence_method}")
        
        method_records.append((answer_text, confidence, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return (max(method_records, key=lambda x: x[1]))

