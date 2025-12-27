import torch
import torch.nn.functional as F
import numpy as np
import math
from collections import Counter

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def calculate_permutation_entropy(time_series, m=3, delay=1):
    n = len(time_series)
    if n < m * delay:
        return 0.0 
    vectors = np.array([time_series[i : i + m * delay : delay] 
                        for i in range(n - (m - 1) * delay)])
    
    if len(vectors) == 0:
        return 0.0
    
    patterns = [tuple(np.argsort(v)) for v in vectors]

    count = Counter(patterns)
    num_vectors = len(patterns)
    probs = np.array([c / num_vectors for c in count.values()])

    pe = -np.sum(probs * np.log2(probs + 1e-9))

    pe_normalized = pe / np.log2(math.factorial(m))

    return pe_normalized


def permutation_entropy(sample_paths, tokenizer, config):

    method_records = []

    PE_DIM = 3  
    PE_DELAY = 1
    alpha = 0 
    beta = 1.0

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        probs = F.softmax(output_scores, dim=-1)
        token_entropies = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            valid_entropies = token_entropies[mask]
        else:
            valid_entropies = token_entropies

        total_entropy = valid_entropies.sum()
        normalization_factor = len(valid_entropies)
        shannon_score = (total_entropy / normalization_factor).item()

        entropy_sequence = valid_entropies.detach().cpu().numpy()
        pe_score = calculate_permutation_entropy(
            entropy_sequence, 
            m=PE_DIM, 
            delay=PE_DELAY
        )

        final_score = (alpha * shannon_score) + (beta * pe_score)

        method_records.append((
            answer_text, 
            final_score, 
            final_answer
        ))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in permutation_entropy")
    
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