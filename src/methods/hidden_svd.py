import numpy as np
import torch

from src.utils import aggregate_paths_based_on_scores_using_min


def get_hidden_states(generated_ids, model):

    hidden_states = None
    
    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states  # tuple: (num_layers, batch, seq, dim)
    
    if hidden_states is None:
        return None

    return hidden_states


def centered_svd_val(Z, alpha=0.001):
    J = torch.eye(Z.shape[0]) - (1 / Z.shape[0]) * torch.ones(Z.shape[0], Z.shape[0])
    Sigma = torch.matmul(torch.matmul(Z.t(), J), Z)
    Sigma = Sigma + alpha * torch.eye(Sigma.shape[0])
    svdvals = torch.linalg.svdvals(Sigma)
    eigscore = torch.log(svdvals).mean()
    return eigscore


def get_svd_eval(hidden_states, layer_num, prompt_len, generated_len, use_len=True):
    svd_score = 0.0
    Z = hidden_states[layer_num]

    if use_len:
        Z = Z[prompt_len: generated_len, :]
    
    Z = torch.transpose(Z, 0, 1)
    svd_score = centered_svd_val(Z).item()

    return svd_score


def compute_hidden_svd(hidden_states, prompt_len, generated_len):
    scores = []
    for layer_num in range(1, len(hidden_states)):
        score = get_svd_eval(hidden_states, layer_num, prompt_len, generated_len, True)
        scores.append(score)

    return np.mean(scores).item()


def hidden_svd(sample_paths, model, tokenizer, config):
    
    method_records = []

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        prompt_len = path["prompt_len"]
        answer_ids = path["answer_ids"]

        if tokenizer.pad_token_id is not None:
            mask = generated_ids != tokenizer.pad_token_id
            generated_ids = generated_ids[mask]

            mask = answer_ids != tokenizer.pad_token_id
            answer_ids = answer_ids[mask]

            prompt_len = len(generated_ids) - len(answer_ids)

        hidden_states = get_hidden_states(generated_ids, model)
        hidden_states = [x[0].to(torch.float32).detach().cpu() for x in hidden_states]

        confidence = compute_hidden_svd(hidden_states, prompt_len, len(generated_ids))

        method_records.append((answer_text, confidence, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")

    if config.aggregate:
        return aggregate_paths_based_on_scores_using_min(method_records)
    else:
        return (min(method_records, key=lambda x: x[1]))

