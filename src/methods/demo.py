from typing import List, Dict, Tuple
import numpy as np
import torch
from src.utils import aggregate_paths_based_on_scores


def get_attn_weights(generated_ids, model):
    """提取注意力权重"""
    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            output_attentions=True,
            return_dict=True
        )
        attn_weights = getattr(outputs, 'attentions', None)
    if attn_weights is None:
        return None
    return attn_weights  # tuple: (num_layers, batch, num_heads, seq, seq)


# ====== 工具函数 ======

def normalize_attention(attn_vec: np.ndarray) -> np.ndarray:
    attn_vec = np.maximum(attn_vec, 0)
    s = attn_vec.sum()
    if s == 0:
        return np.zeros_like(attn_vec)
    return attn_vec / s

def attention_entropy(attn_vec: np.ndarray) -> float:
    p = normalize_attention(attn_vec)
    return -np.sum(p * np.log(p + 1e-12))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ====== 指标实现 ======

def calc_attention_focus(all_attn_maps, prompt_len, seq_len) -> float:
    """注意力集中度（平均熵，越低越好）"""
    entropies = []
    for (_, _), full_attn in all_attn_maps.items():
        for t in range(prompt_len, seq_len):
            attn_to_problem = full_attn[t, :prompt_len]
            entropies.append(attention_entropy(attn_to_problem))
    return float(np.mean(entropies))

def calc_attention_rank(all_attn_maps, prompt_len, seq_len, delta=1e-5) -> float:
    """注意力矩阵秩（越低越好）"""
    rows = []
    for (_, _), full_attn in all_attn_maps.items():
        for t in range(prompt_len, seq_len):
            rows.append(full_attn[t, :prompt_len])
    A = np.stack(rows, axis=0)
    _, S, _ = np.linalg.svd(A, full_matrices=False)
    return float(np.sum(S > delta))

def calc_multihead_consistency(all_attn_maps, prompt_len, seq_len) -> float:
    """多头一致性（越高越好）"""
    dists = []
    for (_, _), full_attn in all_attn_maps.items():
        avg_dist = full_attn[prompt_len:seq_len, :prompt_len].mean(axis=0)
        dists.append(avg_dist)
    sims = []
    for i in range(len(dists)):
        for j in range(i+1, len(dists)):
            sims.append(cosine_similarity(dists[i], dists[j]))
    return float(np.mean(sims)) if sims else 0.0

def calc_attention_stability(all_attn_maps, prompt_len, seq_len, stage_size=5) -> float:
    """跨阶段稳定性（越高越好）"""
    # 取最后一层最后一个head的注意力
    max_layer = max(l for (l, h) in all_attn_maps.keys())
    max_head = max(h for (l, h) in all_attn_maps.keys())
    ref_attn = all_attn_maps[(max_layer, max_head)]
    stage_dists = []
    for start in range(prompt_len, seq_len, stage_size):
        end = min(seq_len, start+stage_size)
        avg_dist = ref_attn[start:end, :prompt_len].mean(axis=0)
        stage_dists.append(avg_dist)
    sims = []
    for i in range(len(stage_dists)-1):
        sims.append(cosine_similarity(stage_dists[i], stage_dists[i+1]))
    return float(np.mean(sims)) if sims else 0.0

def calc_key_token_coverage(all_attn_maps, prompt_len, seq_len, key_token_indices) -> float:
    """关键token覆盖率（越高越好）"""
    covers = []
    for (_, _), full_attn in all_attn_maps.items():
        for t in range(prompt_len, seq_len):
            covers.append(full_attn[t, key_token_indices].sum())
    return float(np.mean(covers)) if covers else 0.0

def calc_anomaly_attention(all_attn_maps, prompt_len, seq_len, anomaly_token_indices) -> float:
    """异常token关注度（越低越好）"""
    scores = []
    for (_, _), full_attn in all_attn_maps.items():
        for t in range(prompt_len, seq_len):
            scores.append(full_attn[t, anomaly_token_indices].sum())
    return float(np.mean(scores)) if scores else 0.0


# ====== 主函数 ======

def attention_enhanced(sample_paths, method_cfg, model, tokenizer, config):
    method_records = []
    delta_rank = method_cfg.get("delta_rank", 1e-5)

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        prompt_len = path["prompt_len"]
        prompt = path["prompt"]

        # 1. 提取注意力权重
        attn_weights = get_attn_weights(generated_ids, model)
        if attn_weights is None:
            raise RuntimeError("No attention weights from model.")
        num_layers = len(attn_weights)
        num_heads = attn_weights[0].shape[1]
        seq_len = len(generated_ids)

        # 2. 保存所有 layer-head 的注意力矩阵
        all_attn_maps = {
            (layer_idx, head_idx): attn_weights[layer_idx][0, head_idx].float().cpu().numpy()
            for layer_idx in range(num_layers)
            for head_idx in range(num_heads)
        }

        # 3. 定关键/异常token集合
        prompt_tokens = tokenizer.tokenize(prompt)
        key_token_indices = [i for i in range(prompt_len) if prompt_tokens[i].isdigit()]
        anomaly_token_indices = [i for i in range(prompt_len) if 'not' in prompt_tokens[i].lower()]

        # 4. 计算各指标
        focus = calc_attention_focus(all_attn_maps, prompt_len, seq_len)
        rank = calc_attention_rank(all_attn_maps, prompt_len, seq_len, delta_rank)
        head_cons = calc_multihead_consistency(all_attn_maps, prompt_len, seq_len)
        stability = calc_attention_stability(all_attn_maps, prompt_len, seq_len)
        coverage = calc_key_token_coverage(all_attn_maps, prompt_len, seq_len, key_token_indices)
        anomaly = calc_anomaly_attention(all_attn_maps, prompt_len, seq_len, anomaly_token_indices)

        # 5. 指标融合（简单线性组合，权重可调整）
        score = (-focus) + (-rank) + head_cons + stability + coverage - anomaly

        method_records.append((answer_text, float(score), final_answer))

    # 6. 返回聚合或最佳
    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return max(method_records, key=lambda x: x[1])