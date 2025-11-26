import numpy as np
import torch

from src.utils import aggregate_paths_based_on_scores


def get_attn_weights(generated_ids, model):

    attn_weights = None

    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            output_attentions=True,
            return_dict=True
        )

        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attn_weights = outputs.attentions  # tuple: (num_layers, batch, num_heads, seq, seq)
    
    if attn_weights is None:
        return None
    
    return attn_weights


def attention_dynamic(sample_paths, method_cfg, model, tokenizer, config):
    
    method_records = []

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]
        prompt_len = path["prompt_len"]
        answer_ids = path["answer_ids"]

        if tokenizer.pad_token_id is not None:
            mask = generated_ids != tokenizer.pad_token_id
            generated_ids = generated_ids[mask]

            mask = answer_ids != tokenizer.pad_token_id
            answer_ids = answer_ids[mask]
            output_scores = output_scores[mask]

            prompt_len = len(generated_ids) - len(answer_ids)

        # Get attention weights
        attn_weights = get_attn_weights(generated_ids, model)
        num_layers = len(attn_weights)
        num_heads = attn_weights[0].shape[1]

        seq_len = len(generated_ids)
        generated_token_indices = range(prompt_len, seq_len)

        d_matrix = np.zeros((num_layers, num_heads))
        all_attn_maps = {}

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                full_attn = attn_weights[layer_idx][0, head_idx].to(torch.float).cpu().numpy()
                all_attn_maps[(layer_idx, head_idx)] = full_attn

                backward_distances = []
                for t in generated_token_indices:
                    p = full_attn[t, :t+1]
                    p = p / p.sum()

                    distances = np.arange(t, -1, -1)

                    weighted_dist = np.sum(p * distances)
                    backward_distances.append(weighted_dist)
                    
                d_matrix[layer_idx, head_idx] = np.mean(backward_distances)

        # Local and global attention map
        flattened_d = d_matrix.flatten()
        num_heads_total = len(flattened_d)

        sorted_indices = np.argsort(flattened_d)
        bottom_k = int(0.3 * num_heads_total)
        top_k = int(0.3 * num_heads_total)

        local_heads_idx = sorted_indices[:bottom_k]
        global_heads_idx = sorted_indices[-top_k:]

        H_loc = [(idx // num_heads, idx % num_heads) for idx in local_heads_idx]
        H_glob = [(idx // num_heads, idx % num_heads) for idx in global_heads_idx]

        A_loc_sum = np.zeros_like(next(iter(all_attn_maps.values())))

        for (l, h) in H_loc:
            A_loc_sum += all_attn_maps[(l, h)]
            A_loc_avg = A_loc_sum / len(H_loc)

        A_glob_sum = np.zeros_like(next(iter(all_attn_maps.values())))
        for (l, h) in H_glob:
            A_glob_sum += all_attn_maps[(l, h)]
            A_glob_avg = A_glob_sum / len(H_glob)

        # WAAD
        window_size = 10
        gen_matrix_loc = A_loc_avg[prompt_len:, prompt_len:]
        seq_gen = gen_matrix_loc.shape[0]
        waad_values = np.zeros(seq_gen)

        for t in range(seq_gen):
            for s in range(t+1):
                dist = min(t - s, window_size)
                waad_values[t] += gen_matrix_loc[t, s] * dist

        # FAI
        H_lo = 10
        H_hi = 50
        gen_matrix_glob = A_glob_avg[prompt_len:, prompt_len:]
        fai_values = np.zeros(seq_gen)

        for s in range(seq_gen):
            start_t = s + H_lo
            end_t = min(seq_gen, s + H_hi)
            if start_t < end_t:
                t_range = range(start_t, end_t)
                fai_values[s] = np.mean([gen_matrix_glob[t, s] for t in t_range])
            else:
                fai_values[s] = 0.0

        delta_t = np.abs(waad_values[:-1] - waad_values[1:])
        q_loc = 0.4
        threshold = np.quantile(delta_t, 1 - q_loc)
        T_loc_indices = np.where(delta_t >= threshold)[0]

        q_glob = 0.4
        threshold_glob = np.quantile(fai_values, 1 - q_glob)
        T_glob_indices = np.where(fai_values >= threshold_glob)[0]
        
        union = np.union1d(T_loc_indices, T_glob_indices)
        # intersection = np.intersect1d(T_loc_indices, T_glob_indices)

        vocab_size = output_scores[0].shape[-1]
        log_vocab_size = np.log(vocab_size)
        confidences = []
        for gen_idx in union:
            logits = output_scores[gen_idx]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            confidence = 1.0 - (entropy / log_vocab_size)
            confidences.append(confidence)
        final_confidence = np.mean(confidences).item() if confidences else 0.0

        method_records.append((answer_text, final_confidence, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return (max(method_records, key=lambda x: x[1]))

