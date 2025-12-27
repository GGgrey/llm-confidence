import torch

from src.utils.utils import aggregate_paths_based_on_scores


def get_hidden_states(generated_ids, model, last_n_layers=None):
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

            if last_n_layers is not None:
                total_layers = len(outputs.hidden_states)
                start_idx = max(0, total_layers - last_n_layers)
                hidden_states = outputs.hidden_states[start_idx:]
            else:
                hidden_states = outputs.hidden_states

        del outputs

    if hidden_states is None:
        return None
    return hidden_states


def compute_layer_spectral_concentration(H, top_k=4):
    H = H.to(torch.float32)
    if H.shape[0] < 2:
        return 0.0
    
    singular_values = torch.linalg.svdvals(H)
    if singular_values.numel() == 0:
        return 0.0
    
    sv_sq = singular_values ** 2
    total_energy = torch.sum(sv_sq)
    if total_energy <= 0:
        return 0.0
    
    k = min(top_k, sv_sq.numel())
    top_energy = torch.sum(sv_sq[:k])

    alpha = (top_energy / (total_energy + 1e-12)).item()
    return alpha


def compute_anisotropy_evolution_score(layer_alphas, 
                                       lambda_trend=0.5, 
                                       mu_osc=0.5):
    if len(layer_alphas) == 0:
        return 0.0
    if len(layer_alphas) == 1:
        return layer_alphas[0]
    
    alphas = torch.tensor(layer_alphas, dtype=torch.float32)
    diffs = alphas[1:] - alphas[:-1]

    A_final = alphas[-1].item()
    T = torch.mean(diffs).item()
    O = torch.mean(torch.abs(diffs)).item()

    score = A_final + lambda_trend * T - mu_osc * O
    return score


def anisotropy_evolution(sample_paths, model, tokenizer, config,
                         last_n_layers=10,
                         top_k=5,
                         lambda_trend=0.5,
                         mu_osc=0.5):
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

        selected_hidden_states = get_hidden_states(generated_ids, model, last_n_layers)
        if selected_hidden_states is None:
            raise RuntimeError("Get hidden states failed")

        prompt_len = len(generated_ids) - len(answer_ids)
        layer_alphas = []

        for layer_tensor in selected_hidden_states:
            layer_hidden = layer_tensor[0].detach().cpu()

            if 0 <= prompt_len < layer_hidden.shape[0]:
                H_answer = layer_hidden[prompt_len:, :]
            else:
                H_answer = layer_hidden

            alpha_l = compute_layer_spectral_concentration(H_answer, top_k=top_k)
            layer_alphas.append(alpha_l)

        score = compute_anisotropy_evolution_score(
            layer_alphas,
            lambda_trend=lambda_trend,
            mu_osc=mu_osc
        )

        method_records.append((answer_text, score, final_answer))

        del selected_hidden_states
        torch.cuda.empty_cache()

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in anisotropy_evolution")
    
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
            
