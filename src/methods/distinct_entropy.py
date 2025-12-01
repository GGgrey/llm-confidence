import torch
import torch.nn.functional as F
from collections import defaultdict

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def _get_ngram_counts(sample_paths, n=3):

    global_counts = defaultdict(int)
    
    for path in sample_paths:
        ids_list = path["answer_ids"].tolist()

        unique_ngrams_in_path = set()
        if len(ids_list) >= n:
            for i in range(len(ids_list) - n + 1):
                ngram = tuple(ids_list[i : i + n])
                unique_ngrams_in_path.add(ngram)
        
        for ngram in unique_ngrams_in_path:
            global_counts[ngram] += 1
            
    return global_counts


def distinct_entropy(sample_paths, method_cfg, normalized_length, tokenizer, config):

    method_records = []
    num_paths = len(sample_paths)

    ngram_n_config = method_cfg.get("distinct_ngram_n", 3)
    if isinstance(ngram_n_config, int):
        ngram_n_list = [ngram_n_config]
    else:
        ngram_n_list = ngram_n_config  # e.g., [2, 3, 4, 5, ...]
    common_threshold_ratio = method_cfg.get("distinct_threshold_ratio", 0.6)

    global_ngram_counts_map = {}
    threshold_count = 0
    
    if num_paths >= 2:
        for n in ngram_n_list:
            global_ngram_counts_map[n] = _get_ngram_counts(sample_paths, n=n)
        threshold_count = num_paths * common_threshold_ratio

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]        # shape: [seq_len]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]  # shape: [seq_len, vocab_size]

        probs = F.softmax(output_scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        valid_mask = torch.ones_like(answer_ids, dtype=torch.bool)
        if tokenizer.pad_token_id is not None:
            valid_mask = valid_mask & (answer_ids != tokenizer.pad_token_id)

        if num_paths >= 2 and len(global_ngram_counts_map) > 0:
            ids_list = answer_ids.tolist()
            distinct_mask = torch.ones_like(answer_ids, dtype=torch.bool)

            for n in ngram_n_list:
                current_counts = global_ngram_counts_map.get(n)
                if not current_counts:
                    continue
                    
                if len(ids_list) >= n:
                    for i in range(len(ids_list) - n + 1):
                        ngram = tuple(ids_list[i : i + n])
                        if current_counts.get(ngram, 0) >= threshold_count:
                            distinct_mask[i : i + n] = False
            
            valid_mask = valid_mask & distinct_mask
        
        filtered_entropy = entropy[valid_mask]

        if len(filtered_entropy) == 0:
             pe = 0.0
        else:
            total_entropy = filtered_entropy.sum()
            current_len = len(filtered_entropy)
            normalization_factor = current_len if normalized_length else 1.0
            pe = (total_entropy / normalization_factor).item()

        method_records.append((answer_text, pe, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores_using_min(method_records)
    else:
        return min(method_records, key=lambda x: x[1])
