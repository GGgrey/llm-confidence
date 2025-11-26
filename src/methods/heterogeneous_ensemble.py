import torch
import torch.nn.functional as F
from src.utils import aggregate_paths_based_on_scores


def compute_ranks_1based(values, smaller_value_is_better=True):
    indexed_data = list(enumerate(values))
    reverse_sort = not smaller_value_is_better
    indexed_data.sort(key=lambda x: x[1], reverse=reverse_sort)
    n = len(values)
    ranks = [0] * n
    for rank, (original_idx, val) in enumerate(indexed_data):
        ranks[original_idx] = rank + 1
    return ranks
    

def robust_z_score_normalization(values, smaller_is_better=True, eps=1e-8):

    if len(values) == 0:
        return []
    
    tensor_vals = torch.tensor(values, dtype=torch.float32)
    
    if len(values) == 1:
        return [0.0]

    median = torch.median(tensor_vals)
    
    abs_diff = torch.abs(tensor_vals - median)
    mad = torch.median(abs_diff)
    
    scale = mad * 1.4826
    
    if scale < eps:
        scale = torch.std(tensor_vals) + eps
    
    z_scores = (tensor_vals - median) / scale
    
    if smaller_is_better:
        z_scores = -z_scores
        
    return z_scores.tolist()
    

def collect_raw_metrics(sample_paths, tokenizer):
    raw_metrics_entropy = []
    raw_metrics_likelihood = []
    raw_metrics_minprob = []
    method_records_meta = []

    for path in sample_paths:
        answer_ids = path["answer_ids"]
        output_scores = path["output_scores"]

        if output_scores.device.type != 'cpu':
            output_scores = output_scores.cpu()
        if answer_ids.device.type != 'cpu':
            answer_ids = answer_ids.cpu()

        probs = F.softmax(output_scores, dim=-1)
        log_probs = F.log_softmax(output_scores, dim=-1)

        token_probs = probs.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1)
        token_log_probs = log_probs.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1)
        entropy_seq = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            valid_entropy = entropy_seq[mask]
            valid_logprobs = token_log_probs[mask]
            valid_probs = token_probs[mask]
        else:
            valid_entropy = entropy_seq
            valid_logprobs = token_log_probs
            valid_probs = token_probs

        length = valid_entropy.size(0)
        if length == 0:
            raw_metrics_entropy.append(1e5)
            raw_metrics_likelihood.append(-1e5)
            raw_metrics_minprob.append(0.0)
        else:
            raw_metrics_entropy.append(valid_entropy.mean().item())
            raw_metrics_likelihood.append(valid_logprobs.mean().item())
            raw_metrics_minprob.append(valid_probs.min().item())

        method_records_meta.append(path)

    metrics = {
        "mean_entropy": raw_metrics_entropy,
        "mean_logprob": raw_metrics_likelihood,
        "min_prob": raw_metrics_minprob
    }
    return metrics, method_records_meta


def ensemble_rrf(metrics, meta, rrf_k=20):
    ranks_e = compute_ranks_1based(metrics["mean_entropy"], smaller_value_is_better=True)
    ranks_l = compute_ranks_1based(metrics["mean_logprob"], smaller_value_is_better=False)
    ranks_m = compute_ranks_1based(metrics["min_prob"], smaller_value_is_better=False)

    records = []
    for i in range(len(meta)):
        rrf_score = (1.0 / (rrf_k + ranks_e[i])) + \
                    (1.0 / (rrf_k + ranks_l[i])) + \
                    (1.0 / (rrf_k + ranks_m[i]))
        records.append((meta[i]["answer_text"], rrf_score, meta[i]["final_answer"]))
    return records


def ensemble_erf(metrics, meta, weight_by="std_inverse"):
    metric_names = ["mean_entropy", "mean_logprob", "min_prob"]
    ranks = {}
    ranks["mean_entropy"] = compute_ranks_1based(metrics["mean_entropy"], True)
    ranks["mean_logprob"] = compute_ranks_1based(metrics["mean_logprob"], False)
    ranks["min_prob"] = compute_ranks_1based(metrics["min_prob"], False)

    weights = {}
    if weight_by == "equal":
        for m in metric_names: weights[m] = 1.0
    else:
        eps = 1e-8
        stds = {}
        for m in metric_names:
            vals = torch.tensor(metrics[m], dtype=torch.float32)
            if vals.numel() <= 1:
                stds[m] = 0.0
            else:
                stds[m] = vals.std(unbiased=True).item()
        
        invs = {}
        for m in metric_names:
            val = stds[m]
            invs[m] = 1.0 / (val + eps) 
        
        total_w = sum(invs.values()) + eps
        for m in metric_names:
            weights[m] = invs[m] / total_w
    
    records = []
    for i in range(len(meta)):
        rank_weighted_sum = 0.0
        for m in metric_names:
            rank_weighted_sum += weights[m] * ranks[m][i]

        score = -rank_weighted_sum
        records.append((meta[i]["answer_text"], score, meta[i]["final_answer"]))
    return records


def ensemble_rsf(metrics, meta, custom_weights=None):
    if custom_weights is None:
        weights = {'mean_entropy': 0.2, 'mean_logprob': 0.6, 'min_prob': 0.2}
    else:
        weights = custom_weights

    z_e = robust_z_score_normalization(metrics["mean_entropy"], smaller_is_better=True)
    z_l = robust_z_score_normalization(metrics["mean_logprob"], smaller_is_better=False)
    z_m = robust_z_score_normalization(metrics["min_prob"], smaller_is_better=False)

    records = []
    for i in range(len(meta)):
        composite_score = (weights['mean_entropy'] * z_e[i]) + \
                          (weights['mean_logprob'] * z_l[i]) + \
                          (weights['min_prob'] * z_m[i])
        records.append((meta[i]["answer_text"], composite_score, meta[i]["final_answer"]))
    return records


def heterogeneous_ensemble(sample_paths, method_cfg, tokenizer, config, mode="rsf", rrf_k=20, weighting_mode="std_inverse", rsf_weights=None):

    raw_metrics, method_records_meta = collect_raw_metrics(sample_paths, tokenizer)
    if not method_records_meta:
        raise ValueError("No valid sample paths provided to heterogeneous ensemble")
    
    mode = method_cfg.get("rank_mode", mode)

    if mode == "rrf":
        rrf_k = method_cfg.get("rrf_k", rrf_k)
        records = ensemble_rrf(raw_metrics, method_records_meta, rrf_k)
    elif mode == "erf":
        weighting_mode = method_cfg.get("weighting_mode", weighting_mode)
        records = ensemble_erf(raw_metrics, method_records_meta, weight_by=weighting_mode)
    elif mode == "rsf":
        rsf_weights = method_cfg.get("rsf_weights", rsf_weights)
        records = ensemble_rsf(raw_metrics, method_records_meta, custom_weights=rsf_weights)
    else:
        raise ValueError(f"Unknown heterogeneous ensemble mode: {mode}")
    
    if not records:
        raise RuntimeError(f"heterogeneous ensemble (mode={mode}) produced no records")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores(records)
    else:
        return max(records, key=lambda x: x[1])