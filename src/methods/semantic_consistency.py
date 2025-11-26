import collections

from src.utils import aggregate_paths_based_on_scores_using_min


def get_ngrams(text, n):
    if not text:
        return set()
    normalized = text.lower().strip()
    if len(normalized) < n:
        return set([normalized])
    return set(normalized[i:i+n] for i in range(len(normalized) - n + 1))


def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    union_len = len(set_a.union(set_b))
    if union_len == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / union_len


def semantic_consistency(sample_paths, config, check_top_n=None):

    method_records = []

    cluster_map = collections.defaultdict(list)
    path_ngrams_cache = []
    ngram_n = 4

    for idx, path in enumerate(sample_paths):
        final_answer = path["final_answer"]
        cluster_map[final_answer].append(idx)

        answer_text = path["answer_text"]
        path_ngrams_cache.append(get_ngrams(answer_text, ngram_n))
    
    path_costs = [0.5] * len(sample_paths)
    for answer, indices in cluster_map.items():
        cluster_size = len(indices)
        if cluster_size == 1:
            path_costs[indices[0]] = 0.6
            continue

        for i in range(cluster_size):
            current_path_idx = indices[i]
            similarity_sum = 0.0
            comparison_count = 0
            for j in range(cluster_size):
                if i == j:
                    continue
            
                peer_idx = indices[j]
                
                sim = jaccard_similarity(
                    path_ngrams_cache[current_path_idx], 
                    path_ngrams_cache[peer_idx]
                )

                similarity_sum += sim
                comparison_count += 1
        
            if comparison_count > 0:
                avg_similarity = similarity_sum / comparison_count
                path_costs[current_path_idx] = 1.0 - avg_similarity
            else:
                path_costs[current_path_idx] = 0.5

    for i, path in enumerate(sample_paths):
        method_records.append((path["answer_text"], path_costs[i], path["final_answer"]))

    if not method_records:
         return ("", 0.0, "")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores_using_min(method_records)
    else:
        return min(method_records, key=lambda x: x[1])
    
