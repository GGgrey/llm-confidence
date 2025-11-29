from abc import ABC, abstractmethod
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
}


def get_metrics_from_config(method_cfg: Dict) -> List[BaseMetric]:
    metric_names = method_cfg.get("ensemble_metrics")
    if not metric_names:
        metric_names = ["mean_entropy", "mean_logprob", "min_prob"]
    
    active_metrics = []
    for name in metric_names:
        if name.startswith("quantile_logit_"):
            try:
                suffix = name.split("_")[-1]
                val = float(suffix)
                alpha = val / 100.0
                alpha = max(0.0, min(1.0, alpha))
                active_metrics.append(QuantileLogitMetric(name=name, alpha=alpha))
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


def compute_ranks(
    values: List[Any],
    smaller_value_is_better: bool = True
) -> List[int]:
    indexed_data = list(enumerate(values))
    reverse_sort = not smaller_value_is_better
    indexed_data.sort(key=lambda x: x[1], reverse=reverse_sort)
    n = len(values)
    ranks = [0] * n
    for rank, (original_idx, val) in enumerate(indexed_data):
        ranks[original_idx] = rank + 1
    return ranks
    

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

        # Calculate logprobs and probs
        log_probs_all = F.log_softmax(output_scores, dim=-1, dtype=torch.float64)
        probs_all = F.softmax(output_scores, dim=-1, dtype=torch.float64)
        
        token_idx = answer_ids.unsqueeze(-1)
        token_log_probs = log_probs_all.gather(dim=-1, index=token_idx).squeeze(-1)
        token_probs = probs_all.gather(dim=-1, index=token_idx).squeeze(-1)
        token_logits = output_scores.gather(dim=-1, index=token_idx).squeeze(-1)

        # Calculate entropy
        entropy_seq = -torch.sum(probs_all * torch.log(probs_all + 1e-9), dim=-1, dtype=torch.float64)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            entropy_seq = entropy_seq[mask]
            token_log_probs = token_log_probs[mask]
            token_probs = token_probs[mask]
            token_logits = token_logits[mask]
        context = {
            "entropy": entropy_seq,
            "logprobs": token_log_probs,
            "probs": token_probs,
            "logits": token_logits,
            "raw_output_scores": output_scores,
            "raw_answer_ids": answer_ids
        }

        for metric in active_metrics:
            try:
                val = metric.compute(context)
            except Exception as e:
                raise RuntimeError(f"Metric {metric.name} computation failed: {e}")
            metrics_results[metric.name].append(val)
        
        method_records.append(path)
    
    return metrics_results, method_records


def ensemble_rrf(
    metrics_data: Dict[str, List[float]], 
    paths: List[Dict], 
    active_metrics: List[BaseMetric],
    rrf_k: int = 20
):
    num_samples = len(paths)
    all_ranks = {}
    for metric in active_metrics:
        raw_values = metrics_data[metric.name]
        all_ranks[metric.name] = compute_ranks(
            raw_values, 
            smaller_value_is_better=metric.lower_is_better
        )
    
    records = []
    for i in range(num_samples):
        rrf_score = 0.0
        for metric in active_metrics:
            rank_i = all_ranks[metric.name][i]
            rrf_score += 1.0 / (rrf_k + rank_i)
        
        records.append((
            paths[i].get("answer_text", ""), 
            rrf_score, 
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
    weighting_mode="std_inverse"
):
    if not sample_paths:
        raise ValueError("Sample paths list is empty")
    
    active_metrics = get_metrics_from_config(method_cfg)
    
    raw_metrics_data, method_records = collect_metrics_data(
        sample_paths, tokenizer, active_metrics
    )
    if not method_records:
        raise ValueError("No valid sample paths provided to heterogeneous ensemble")
    
    mode = method_cfg.get("rank_mode", mode)
    rrf_k = method_cfg.get("rrf_k", rrf_k)

    if mode == "rrf":
        records = ensemble_rrf(
            raw_metrics_data, method_records, active_metrics, rrf_k=rrf_k
        )
    elif mode == "xxx":  # Todo
        pass
    else:
        raise ValueError(f"Unknown heterogeneous ensemble mode: {mode}")
    
    if not records:
        raise RuntimeError(f"Heterogeneous ensemble (mode={mode}) produced no records")
    
    if hasattr(config, 'aggregate') and config.aggregate:
        return aggregate_paths_based_on_scores(records)
    else:
        return max(records, key=lambda x: x[1])