import torch
import torch.nn.functional as F
import numpy as np
import math

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def get_hidden_states(generated_ids, model):
    hidden_states = None

    if generated_ids.device != model.device:
        generated_ids = generated_ids.to(model.device)
        
    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True
        )

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states

        del outputs

    if hidden_states is None:
        return None
    return hidden_states


def compute_vector_permutation_entropy(vector, m=3, tau=1, denom=None):
    vector = vector.to(torch.float32)
    n = vector.shape[0]

    if n < m:
        return 1.0

    windows = vector.unfold(0, m, tau)
    
    sorted_idx = torch.argsort(windows, dim=1)
    
    sorted_idx_np = sorted_idx.cpu().numpy()
    
    _, counts = np.unique(sorted_idx_np, axis=0, return_counts=True)
    
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log2(probs + 1e-12))
    
    if denom is None:
        denom = np.log2(math.factorial(m))
        
    return pe / (denom + 1e-12)


def compute_sequence_complexity(hidden_states, m=4, tau=1):
    if hidden_states.numel() == 0:
        return 1.0
        
    T = hidden_states.shape[0]
    
    norm_denom = np.log2(math.factorial(m))
    
    total_pe = 0.0
    valid_steps = 0

    for t in range(T):
        vec = hidden_states[t]
        pe = compute_vector_permutation_entropy(vec, m=m, tau=tau, denom=norm_denom)
        total_pe += pe
        valid_steps += 1
        
    if valid_steps == 0:
        return 1.0
        
    return total_pe / valid_steps


def hidden_complexity(sample_paths, model, tokenizer, config, m_dim=4, tau=1):

    method_records = []

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        answer_ids = path["answer_ids"]

        if tokenizer.pad_token_id is not None:
            mask = generated_ids != tokenizer.pad_token_id
            generated_ids = generated_ids[mask]
            mask = answer_ids != tokenizer.pad_token_id
            answer_ids = answer_ids[mask]

        hidden_states = get_hidden_states(generated_ids, model)
        if hidden_states is None: 
            raise RuntimeError("Get hidden states failed")
        
        last_layer_hidden = hidden_states[-1][0].detach().cpu()

        prompt_len = len(generated_ids) - len(answer_ids)
        H = last_layer_hidden[prompt_len:, :]

        avg_pe_score = compute_sequence_complexity(
            H, 
            m=m_dim, 
            tau=tau
        )

        score = -avg_pe_score
        
        method_records.append((answer_text, score, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in hidden_complexity")
    
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