import numpy as np
import torch

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


def compute_stable_rank(hidden_states_matrix):
    H = hidden_states_matrix.to(torch.float32)

    # Compute singular values 
    singular_values = torch.linalg.svdvals(H)
    max_singular_value = singular_values[0]

    numerator = torch.sum(singular_values ** 2)
    denominator = max_singular_value ** 2

    if denominator == 0:
        return 0.0
    
    sr_score = numerator / denominator
    return sr_score.item()


def stable_rank(sample_paths, model, tokenizer, config):

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

        # Extract all layers' hidden states
        all_hidden_states = get_hidden_states(generated_ids, model)

        # Select the last layer
        last_layer_hidden = all_hidden_states[-1][0].detach().cpu()

        # Only retain output tokens' hidden states
        prompt_len = len(generated_ids) - len(answer_ids)
        H = last_layer_hidden[prompt_len:, :]

        # Compute stable rank
        score = compute_stable_rank(H)

        method_records.append((answer_text, score, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in stable_rank")
    
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