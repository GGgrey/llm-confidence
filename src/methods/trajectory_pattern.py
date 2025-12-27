import torch
import numpy as np

from src.utils.utils import aggregate_paths_based_on_scores


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


def compute_trajectory_score(hidden_states):
    H = hidden_states.to(torch.float32)
    T, d = H.shape

    if T < 3:
        return 0.0
    
    velocities = H[1:] - H[:-1]
    norms = torch.norm(velocities, dim=1, keepdim=True) + 1e-8
    normalized_velocities = velocities / norms

    v_curr = normalized_velocities[:-1]
    v_next = normalized_velocities[1:]

    cosine_sims = torch.sum(v_curr * v_next, dim=1)

    local_curvatures = 1.0 - cosine_sims

    mean_curvature = torch.mean(local_curvatures).item()

    return mean_curvature


def trajectory_pattern(sample_paths, model, tokenizer, config):

    method_records = []

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        answer_ids = path["answer_ids"]

        # Filter padding tokens
        if tokenizer.pad_token_id is not None:
            mask = generated_ids != tokenizer.pad_token_id
            generated_ids = generated_ids[mask]

            mask = answer_ids != tokenizer.pad_token_id
            answer_ids = answer_ids[mask]

        hidden_states = get_hidden_states(generated_ids, model)
        if hidden_states is None: 
            raise RuntimeError("Get hidden states failed")
        
        # Use last layer
        last_layer_hidden = hidden_states[-1][0].detach().cpu()

        # Slice for answer only
        prompt_len = len(generated_ids) - len(answer_ids)
        H = last_layer_hidden[prompt_len:, :]

        score = compute_trajectory_score(H)

        method_records.append((answer_text, score, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in trajectory_pattern")
    
    path_info = [
        {"answer_text": a, "score": s, "final_answer": f}
        for (a, s, f) in method_records
    ]

    if config.aggregate:
        result = aggregate_paths_based_on_scores(method_records)
    else:
        result = max(method_records, key=lambda x: x[1])
    
    answer_text, score, final_answer = result
    return answer_text, score, final_answer, path_info
