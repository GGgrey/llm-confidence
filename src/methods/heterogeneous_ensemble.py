from abc import ABC, abstractmethod
from collections import defaultdict
import math
import statistics
from typing import List, Dict, Any, Tuple, Type, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores


class BaseMetric(ABC):
    def __init__(self, name: str, lower_is_better: bool):
        self.name = name
        self.lower_is_better = lower_is_better

    @abstractmethod
    def compute(self, context: Dict[str, Any]) -> float:
        pass


class MeanEntropyMetric(BaseMetric):
    def __init__(self):
        super().__init__("mean_entropy", lower_is_better=True)

    def compute(self, context: Dict[str, Any]) -> float:
        return context["entropy"].mean().item()
    

class MeanLogProbMetric(BaseMetric):
    def __init__(self):
        super().__init__("mean_logprob", lower_is_better=False)

    def compute(self, context: Dict[str, Any]) -> float:
        return context["logprobs"].mean().item()
    

class MinProbMetric(BaseMetric):
    def __init__(self):
        super().__init__("min_prob", lower_is_better=False)

    def compute(self, context: Dict[str, Any]) -> float:
        if context["probs"].numel() == 0:
            return 0.0
        return context["probs"].min().item()
    

class PerplexityMetric(BaseMetric):
    def __init__(self):
        super().__init__("ppl", lower_is_better=True)

    def compute(self, context: Dict[str, Any]) -> float:
        mean_lp = context["logprobs"].mean()
        return torch.exp(-mean_lp).item()
    

class MeanLogitMetric(BaseMetric):
    def __init__(self):
        super().__init__("mean_logit", lower_is_better=False)

    def compute(self, context: Dict[str, Any]) -> float:
        return context["logits"].mean().item()
    

class QuantileLogitMetric(BaseMetric):
    def __init__(self, name: str, alpha: float):
        super().__init__(name, lower_is_better=False)
        self.alpha = alpha

    def compute(self, context: Dict[str, Any]) -> float:
        if context["logits"].numel() == 0:
            return 0.0
        s_quant = torch.quantile(context["logits"], self.alpha).item()
        return s_quant
    

class QuantileEntropyMetric(BaseMetric):
    def __init__(self, name: str, alpha: float):
        super().__init__(name, lower_is_better=True)
        self.alpha = alpha

    def compute(self, context: Dict[str, Any]) -> float:
        if context["entropy"].numel() == 0:
            return 0.0
        s_quant = torch.quantile(context["entropy"], self.alpha).item()
        return s_quant


class LengthMetric(BaseMetric):
    def __init__(self):
        super().__init__("length", lower_is_better=True)

    def compute(self, context: Dict[str, Any]) -> float:
        seq_len = context["entropy"].numel()
        return np.log1p(seq_len)
    

class TrendMetric(BaseMetric):
    def __init__(self):
        super().__init__("trend", lower_is_better=True)

    def compute(self, context: Dict[str, Any]) -> float:
        seq_len = context["entropy"].numel()
        if seq_len < 2:
            return 0.0
        y = context["entropy"].cpu().numpy()
        x = np.arange(seq_len)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    

class TailConfidenceMetric(BaseMetric):
    def __init__(self, tail_tokens: int):
        super().__init__("tail_confidence", lower_is_better=True)
        self.tail_tokens = tail_tokens

    def compute(self, context: Dict[str, Any]) -> float:
        if len(context["entropy"]) <= self.tail_tokens:
            return context["entropy"].mean().item()
        tail_confidence = context["entropy"][-self.tail_tokens:]
        return tail_confidence.mean().item()
    

class GroupConfidence(BaseMetric):
    def __init__(self, name: str, window_size=50, bottom_percent=0.1):
        super().__init__(name, lower_is_better=True)
        self.name = name
        self.window_size = window_size
        self.bottom_percent = bottom_percent

    def compute(self, context: Dict[str, Any]) -> float:
        entropy = context["entropy"]  # Shape: (seq_len,)
        seq_len = entropy.numel()
        if seq_len <= self.window_size: 
            return entropy.mean().item()
        windows = entropy.unfold(dimension=0, size=self.window_size, step=1)
        window_means = windows.mean(dim=1)  # Shape: (num_windows,)
        if window_means.numel() == 0:
            return 0.0
        if self.name == "mean_group_confidence":
            return window_means.mean().item()
        elif self.name == "lowest_group_confidence":
            return window_means.min().item()
        elif self.name == "bottom_group_confidence":
            num_bottom = max(1, int(len(window_means) * self.bottom_percent))
            if num_bottom == 1:
                return window_means.min().item()
            else:
                bottom_means, _ = torch.topk(window_means, k=num_bottom, largest=False)
                return bottom_means.mean().item()
        else:
            raise ValueError(f"Unsupported method: {self.name}")
        

class SelfCertaintyMetric(BaseMetric):
    def __init__(self):
        super().__init__("self_certainty", lower_is_better=False)

    def compute(self, context: Dict[str, Any]) -> float:
        return context["self_certainty"].mean().item()


METRIC_REGISTRY: Dict[str, Any] = {
    "mean_entropy": MeanEntropyMetric,
    "mean_logprob": MeanLogProbMetric,
    "min_prob": MinProbMetric,
    "ppl": PerplexityMetric,
    "mean_logit": MeanLogitMetric,
    "length": LengthMetric,
    "trend": TrendMetric,
    "tail_confidence": lambda: TailConfidenceMetric(tail_tokens=50),
    "lowest_group_confidence": lambda: GroupConfidence(name="lowest_group_confidence", window_size=50),
    "mean_group_confidence": lambda: GroupConfidence(name="mean_group_confidence", window_size=50),
    "bottom_group_confidence": lambda: GroupConfidence(name="bottom_group_confidence", window_size=50, bottom_percent=0.1),
    "self_certainty": SelfCertaintyMetric,
}


def get_metrics_from_config(method_cfg: Dict) -> List[BaseMetric]:
    metric_names = method_cfg.get("ensemble_metrics")
    if not metric_names:
        metric_names = ["mean_entropy", "mean_logprob", "min_prob"]
    
    active_metrics = []
    for name in metric_names:
        if name.startswith("quantile_"):
            try:
                suffix = name.split("_")[-1]
                val = float(suffix)
                alpha = val / 100.0
                alpha = max(0.0, min(1.0, alpha))
                if "logit" in name:
                    active_metrics.append(QuantileLogitMetric(name=name, alpha=alpha))
                elif "entropy" in name:
                    active_metrics.append(QuantileEntropyMetric(name=name, alpha=alpha))
            except ValueError:
                raise ValueError(f"Invalid quantile metric format: {name}")
        elif name in METRIC_REGISTRY:
            try:
                active_metrics.append(METRIC_REGISTRY[name]())
            except TypeError as e:
                raise TypeError(f"Error instantiating metric '{name}': {e}")
        else:
            raise ValueError(
                f"Metric '{name}' not found. "
                f"Available: {list(METRIC_REGISTRY.keys())}"
            )

    return active_metrics


@torch.no_grad()
def collect_metrics_data(
    sample_paths: List[Dict], 
    tokenizer, 
    active_metrics: List[BaseMetric]
) -> Tuple[Dict[str, List[Any]], List[Dict]]:
    
    metrics_results = {m.name: [] for m in active_metrics}
    method_records = []

    for path in sample_paths:
        answer_ids = path["answer_ids"]        # shape: [seq_len]
        output_scores = path["output_scores"]  # shape: [seq_len, vocab_size]
        output_logits = path["output_logits"]

        # output_scores = output_scores.to("cpu")
        # answer_ids = answer_ids.to("cpu")

        # Calculate logprobs and probs
        log_probs_all = F.log_softmax(output_scores, dim=-1)
        probs_all = F.softmax(output_scores, dim=-1)
        
        token_idx = answer_ids.unsqueeze(-1)
        token_log_probs = log_probs_all.gather(dim=-1, index=token_idx).squeeze(-1)
        token_probs = probs_all.gather(dim=-1, index=token_idx).squeeze(-1)
        token_logits = output_scores.gather(dim=-1, index=token_idx).squeeze(-1)

        V = output_logits.size(-1)
        V_tensor = torch.tensor(V, dtype=output_logits.dtype, device=output_logits.device)
        log_probs_from_logits = F.log_softmax(output_logits, dim=-1)
        logprob_sum = torch.sum(log_probs_from_logits, dim=-1)
        self_certainty = -1/V * logprob_sum - torch.log(V_tensor)

        # Calculate entropy
        entropy_seq = -torch.sum(probs_all * torch.log(probs_all + 1e-9), dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            entropy_seq = entropy_seq[mask]
            token_log_probs = token_log_probs[mask]
            token_probs = token_probs[mask]
            token_logits = token_logits[mask]
            self_certainty = self_certainty[mask]
        context = {
            "entropy": entropy_seq,
            "logprobs": token_log_probs,
            "probs": token_probs,
            "logits": token_logits,
            "self_certainty": self_certainty,
        }

        for metric in active_metrics:
            try:
                val = metric.compute(context)
            except Exception as e:
                raise RuntimeError(f"Metric {metric.name} computation failed: {e}")
            metrics_results[metric.name].append(val)
        
        method_records.append(path)
    
    return metrics_results, method_records


class Standardizer:
    @staticmethod
    def min_max_scale(values: List[float], lower_is_better: bool) -> List[float]:
        if not values:
            return []
        
        min_v, max_v = min(values), max(values)
        span = max_v - min_v

        # Prevent division by zero if all values are the same
        if span == 0:
            return [0.5] * len(values)
        
        normalized = []
        for v in values:
            # Standard normalization: (x - min) / (max - min) -> [0, 1]
            score = (v - min_v) / span

            # If the original metric is "lower is better", invert it
            if lower_is_better:
                score = 1.0 - score
            normalized.append(score)
        
        return normalized

    @staticmethod
    def zscore_sigmoid(values: List[float], lower_is_better: bool) -> List[float]:
        if not values:
            return []
        
        n = len(values)
        mu = sum(values) / n
        var = sum((x - mu) ** 2 for x in values) / n
        sigma = math.sqrt(var)

        if sigma == 0:
            return [0.5] * n
        
        normalized = []
        for x in values:
            z = (x - mu) / sigma
            if lower_is_better:
                z = -z
            s = 1.0 / (1.0 + math.exp(-z))
            normalized.append(s)
        
        return normalized
        

def compute_ranks(
    values: List[Any],
    smaller_value_is_better: bool = True
) -> List[int]:
    indexed_values = list(enumerate(values))

    # If smaller_value_is_better (e.g., perplexity), sort ascending
    # Otherwise (e.g., log_prob), sort descending
    sorted_indexed_values = sorted(
        indexed_values, 
        key=lambda x: x[1], 
        reverse=not smaller_value_is_better
    )
    
    ranks = [0] * len(values)

    # Assign ranks (starting from 1)
    for rank, (original_idx, _) in enumerate(sorted_indexed_values, start=1):
        ranks[original_idx] = rank

    return ranks


def ensemble_rrf(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[BaseMetric],
    rrf_k: int = 20,
    p: float = 1.0
):
    num_samples = len(paths)
    all_ranks = {}

    # Compute ranks for all metrics
    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        all_ranks[metric.name] = compute_ranks(
            raw_values, 
            smaller_value_is_better=metric.lower_is_better
        )
    
    records = []

    # Compute fusion scores
    for i in range(num_samples):
        rrf_score = 0.0
        for metric in active_metrics:
            rank_i = all_ranks[metric.name][i]
            # RRF: 1 / ((k + rank) ** p)
            # Default p=1.0 is standard RRF
            denominator = (rrf_k + rank_i) ** p
            rrf_score += 1.0 / denominator
        
        records.append((
            paths[i].get("answer_text", ""), 
            rrf_score, 
            paths[i].get("final_answer")
        ))
    
    return records


def ensemble_borda(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[BaseMetric],
    p: float = 1.0
):
    num_samples = len(paths)
    all_ranks = {}

    # Compute ranks for all metrics
    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        all_ranks[metric.name] = compute_ranks(
            raw_values, 
            smaller_value_is_better=metric.lower_is_better
        )

    records = []

    # Compute fusion scores
    for i in range(num_samples):
        borda_score = 0.0
        for metric in active_metrics:
            rank_i = all_ranks[metric.name][i]
            # Borda base score: N - rank + 1 (1st place gets N points, last gets 1)
            base_score  = num_samples - rank_i + 1
            # Apply hyperparameter p for power scaling
            # Default p=1.0 is standard Borda
            score = base_score ** p
            borda_score += score

        records.append((
            paths[i].get("answer_text", ""), 
            borda_score, 
            paths[i].get("final_answer")
        ))
    
    return records


def ensemble_combsum(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[Any]
):
    num_samples = len(paths)
    normalized_scores = {}

    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        # Perform min-max normalization
        normalized_scores[metric.name] = Standardizer.zscore_sigmoid(
            raw_values, 
            lower_is_better=metric.lower_is_better
        )

    records = []
    for i in range(num_samples):
        total_score = 0.0
        for metric in active_metrics:
            # Get the normalized score for the i-th sample on the current metric
            score_i = normalized_scores[metric.name][i]
            total_score += score_i

        records.append((
            paths[i].get("answer_text", ""), 
            total_score, 
            paths[i].get("final_answer")
        ))
    
    return records


def ensemble_combmnz(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[Any]
) -> List[Tuple[str, float, Any]]:
    num_samples = len(paths)
    normalized_scores = {}

    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        normalized_scores[metric.name] = Standardizer.zscore_sigmoid(
            raw_values, 
            lower_is_better=metric.lower_is_better
        )

    records = []
    for i in range(num_samples):
        sum_score = 0.0
        non_zero_count = 0

        for metric in active_metrics:
            val = normalized_scores[metric.name][i]
            sum_score += val
            if val > 1e-9:
                non_zero_count += 1

        final_score = sum_score * non_zero_count

        records.append((
            paths[i].get("answer_text", ""), 
            final_score, 
            paths[i].get("final_answer")
        ))
            
    return records


def ensemble_exp_sq(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[Any],
    mode: str = "expcombsum"
) -> List[Tuple[str, float, Any]]:
    num_samples = len(paths)
    normalized_scores = {}

    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        normalized_scores[metric.name] = Standardizer.zscore_sigmoid(
            raw_values, 
            lower_is_better=metric.lower_is_better
        )

    records = []
    for i in range(num_samples):
        transformed_sum = 0.0
        non_zero_count_original = 0

        for metric in active_metrics:
            norm_val = normalized_scores[metric.name][i]

            if mode.startswith("exp"):
                transformed_val = math.exp(norm_val)
            elif mode.startswith("sq"):
                transformed_val = norm_val ** 2
            else:
                raise ValueError(f"Unknown transformation mode: {mode}")
            
            transformed_sum += transformed_val
            if norm_val > 1e-9:
                non_zero_count_original += 1
            
        if "mnz" in mode:
            final_score = transformed_sum * non_zero_count_original
        else:
            final_score = transformed_sum

        records.append((
            paths[i].get("answer_text", ""), 
            final_score, 
            paths[i].get("final_answer")
        ))
    
    return records


def ensemble_combsum_topn(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[Any],
    top_n: int = 5
) -> List[Tuple[str, float, Any]]:
    num_samples = len(paths)
    normalized_scores = {}

    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        normalized_scores[metric.name] = Standardizer.zscore_sigmoid(
            raw_values, 
            lower_is_better=metric.lower_is_better
        )

    path_scores = []
    for i in range(num_samples):
        current_path_sum = 0.0
        for metric in active_metrics:
            current_path_sum += normalized_scores[metric.name][i]
        path_scores.append(current_path_sum)
    
    answer_groups = defaultdict(list)
    for i in range(num_samples):
        key = paths[i].get("final_answer")
        if key is None:
             key = paths[i].get("answer_text", "")
        
        answer_groups[key].append((path_scores[i], i))
    
    records = [None] * num_samples
    for i in range(num_samples):
        records[i] = (paths[i].get("answer_text", ""), 0.0, paths[i].get("final_answer"))

    for key, group_items in answer_groups.items():
        sorted_items = sorted(group_items, key=lambda x: x[0], reverse=True)
        top_items = sorted_items[:top_n]
        group_final_score = sum(item[0] for item in top_items)
        
        for _, idx in group_items:
            records[idx] = (
                paths[idx].get("answer_text", ""), 
                group_final_score, 
                paths[idx].get("final_answer")
            )
    
    return records


def ensemble_stat(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[Any],
    mode: str = "combmax"
) -> List[Tuple[str, float, Any]]:
    num_samples = len(paths)
    normalized_scores = {}

    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        normalized_scores[metric.name] = Standardizer.zscore_sigmoid(
            raw_values, 
            lower_is_better=metric.lower_is_better
        )

    records = []
    for i in range(num_samples):
        scores_for_path = []
        for metric in active_metrics:
            scores_for_path.append(normalized_scores[metric.name][i])
        
        if not scores_for_path:
            final_score = 0.0
        else:
            if mode == "combmax":
                final_score = max(scores_for_path)
            elif mode == "combmin":
                final_score = min(scores_for_path)
            elif mode == "combmed":
                final_score = statistics.median(scores_for_path)
            else:
                raise ValueError(f"Unknown stat mode: {mode}")

        records.append((
            paths[i].get("answer_text", ""), 
            final_score, 
            paths[i].get("final_answer")
        ))
    
    return records


def ensemble_combanz(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[Any]
) -> List[Tuple[str, float, Any]]:
    num_samples = len(paths)
    normalized_scores = {}

    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        normalized_scores[metric.name] = Standardizer.zscore_sigmoid(
            raw_values, 
            lower_is_better=metric.lower_is_better
        )

    records = []

    for i in range(num_samples):
        sum_score = 0.0
        count = 0

        for metric in active_metrics:
            val = normalized_scores[metric.name][i]
            sum_score += val
            count += 1

        if count > 0:
            final_score = sum_score / count
        else:
            final_score = 0.0

        records.append((
            paths[i].get("answer_text", ""), 
            final_score, 
            paths[i].get("final_answer")
        ))
            
    return records


def ensemble_rank_weighted_variants(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[Any],
    mode: str = "rr_x",
    x_power: float = 1.0
) -> List[Tuple[str, float, Any]]:
    num_samples = len(paths)
    normalized_scores = {}
    all_ranks = {}

    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        
        all_ranks[metric.name] = compute_ranks(
            raw_values, 
            smaller_value_is_better=metric.lower_is_better
        )
        
        if mode != "rr_x":
            normalized_scores[metric.name] = Standardizer.zscore_sigmoid(
                raw_values, 
                lower_is_better=metric.lower_is_better
            )

    records = []
    for i in range(num_samples):
        final_score = 0.0

        for metric in active_metrics:
            rank_val = all_ranks[metric.name][i]
            rank_weight = (1.0 / rank_val) ** x_power

            if mode == "rr_x":
                final_score += rank_weight
            elif mode == "combsum_rr_x":
                score_val = normalized_scores[metric.name][i]
                final_score += score_val * rank_weight
            elif mode == "sqcombsum_rr_x":
                score_val = normalized_scores[metric.name][i]
                final_score += (score_val ** 2) * rank_weight
            else:
                raise ValueError(f"Unknown rank weighted mode: {mode}")
        
        records.append((
            paths[i].get("answer_text", ""), 
            final_score, 
            paths[i].get("final_answer")
        ))
    
    return records


def heterogeneous_ensemble(
    sample_paths,
    method_cfg,
    tokenizer,
    config,
    mode="rsf",
    rrf_k=20,
    power_p=1.0,
    weighting_mode="std_inverse"
):
    if not sample_paths:
        raise ValueError("Sample paths list is empty")
    
    active_metrics = get_metrics_from_config(method_cfg)
    
    # Collect raw metric data
    raw_metrics_data, method_records = collect_metrics_data(
        sample_paths, tokenizer, active_metrics
    )
    if not method_records:
        raise ValueError("No valid sample paths provided to heterogeneous ensemble")
    
    mode = method_cfg.get("rank_mode", mode)
    rrf_k = method_cfg.get("rrf_k", rrf_k)
    power_p = method_cfg.get("power_p", 1.0)
    top_n = method_cfg.get("top_n", 5)

    if mode == "rrf":
        records = ensemble_rrf(
            raw_metrics_data, method_records, active_metrics, rrf_k=rrf_k, p=power_p
        )
    elif mode == "borda":
        records = ensemble_borda(
            raw_metrics_data, method_records, active_metrics, p=power_p
        )
    elif mode == "combsum":
        records = ensemble_combsum(
            raw_metrics_data, method_records, active_metrics
        )
    elif mode == "combmnz":
        records = ensemble_combmnz(
            raw_metrics_data, method_records, active_metrics
        )
    elif mode in ["expcombsum", "expcombmnz", "sqcombsum", "sqcombmnz"]:
        records = ensemble_exp_sq(
            raw_metrics_data, method_records, active_metrics, mode=mode
        )
    elif mode == "combsum_topn":
        records = ensemble_combsum_topn(
            raw_metrics_data, method_records, active_metrics, top_n=top_n
        )
    elif mode in ["combmax", "combmin", "combmed"]:
        records = ensemble_stat(
            raw_metrics_data, method_records, active_metrics, mode=mode
        )
    elif mode == "combanz":
        records = ensemble_combanz(
            raw_metrics_data, method_records, active_metrics
        )
    elif mode in ["rr_x", "combsum_rr_x", "sqcombsum_rr_x"]:
        records = ensemble_rank_weighted_variants(
            raw_metrics_data, method_records, active_metrics, mode=mode, x_power=power_p
        )
    else:
        raise ValueError(f"Unknown heterogeneous ensemble mode: {mode}")
    
    if not records:
        raise RuntimeError(f"Heterogeneous ensemble (mode={mode}) produced no records")
    
    path_info = [
        {"answer_text": a, "score": s, "final_answer": f}
        for (a, s, f) in records
    ]
    
    if hasattr(config, "aggregate") and config.aggregate:
        result = aggregate_paths_based_on_scores(records)
    else:
        result = max(records, key=lambda x: x[1])
    
    answer_text, score, final_answer = result
    return answer_text, score, final_answer, path_info