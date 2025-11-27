import numpy as np
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def trend_linear_regression(entropy):
    x = np.arange(len(entropy)).reshape(-1, 1)
    y = np.array(entropy)
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(x, y)
    trend = "upward" if slope > 0 else "downward"
    return slope, intercept, r2, trend


def trend_savgol(entropy, window=7, poly=3):
    smooth_entropy = savgol_filter(entropy, window_length=min(window, len(entropy)), polyorder=poly)
    diff = np.diff(smooth_entropy)
    mean_diff = np.mean(diff)
    trend = "upward" if mean_diff > 0 else "downward"
    return mean_diff, trend, smooth_entropy


def trend_theilsen(entropy):
    x = np.arange(len(entropy)).reshape(-1, 1)
    y = np.array(entropy)
    model = TheilSenRegressor(random_state=0)
    model.fit(x, y)
    slope = model.coef_[0]
    trend = "upward" if slope > 0 else "downward"
    return slope, trend


def trend_estimation(sample_paths, method_cfg, tokenizer, config):
    
    method_records = []
    estimation_method = method_cfg["estimation_method"]

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        probs = F.softmax(output_scores, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            entropy = entropy[mask]

        window_size = 10
        if len(entropy) >= window_size:
            entropy = entropy.unfold(dimension=0, size=window_size, step=1)
            entropy = entropy.mean(dim=-1)

        slope = 1.0
        if estimation_method == "linear_regression":
            slope, _, _, _ = trend_linear_regression(entropy.detach().cpu().numpy())
        elif estimation_method == "savgol":
            slope, _, _ = trend_savgol(entropy.detach().cpu().numpy())
        elif estimation_method == "theilsen":
            slope, _ = trend_theilsen(entropy.detach().cpu().numpy())
        else:
            raise ValueError(f"Unsupported estimation method: {estimation_method}")

        method_records.append((answer_text, slope, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores_using_min(method_records)
    else:
        return (min(method_records, key=lambda x: x[1]))
        

