import torch
import torch.nn.functional as F

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

    return hidden_states


def compute_stable_rank(hidden_states_matrix: torch.Tensor) -> float:
    H = hidden_states_matrix.to(torch.float32)

    singular_values = torch.linalg.svdvals(H)
    max_singular_value = singular_values[0]

    numerator = torch.sum(singular_values ** 2)
    denominator = max_singular_value ** 2

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


def rank_weighted_confidence(
    sample_paths,
    model,
    tokenizer,
    config,
    normalized_length=True,
    top_k=20,
    window_size=10,
    eps=1e-9,
):
    method_records = []

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]

        generated_ids = path["generated_ids"]
        answer_ids = path["answer_ids"]
        output_scores = path["output_scores"]

        if tokenizer.pad_token_id is not None:
            mask = generated_ids != tokenizer.pad_token_id
            generated_ids = generated_ids[mask]

            mask = answer_ids != tokenizer.pad_token_id
            answer_ids = answer_ids[mask]

        if top_k is None:
            probs = F.softmax(output_scores, dim=-1)
        else:
            top_values = torch.topk(output_scores, k=top_k, dim=-1).values
            probs = F.softmax(top_values, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)

        if tokenizer.pad_token_id is not None:
            entropy = entropy[mask]

        if len(entropy) < window_size:
            total_entropy = entropy.sum()
            normalization_factor = len(entropy) if normalized_length else 1.0
            score = (total_entropy / normalization_factor).item()
            method_records.append((answer_text, score, final_answer))
            continue

        all_hidden_states = get_hidden_states(generated_ids, model)
        if all_hidden_states is None:
            raise RuntimeError("Get hidden states failed")
        
        last_layer_hidden = all_hidden_states[-1][0].detach().cpu()
        prompt_len = len(generated_ids) - len(answer_ids)
        H = last_layer_hidden[prompt_len:, :]

        if H.shape[0] != entropy.shape[0]:
            raise ValueError("Shape not match")
        
        entropy_windows = entropy.unfold(dimension=0, size=window_size, step=1)
        window_ent_means = entropy_windows.mean(dim=-1)

        H_windows = H.unfold(dimension=0, size=window_size, step=1)

        weights = []
        for i in range(H_windows.shape[0]):
            w = compute_stable_rank(H_windows[i])
            weights.append(w)
        weights = torch.tensor(weights, dtype=torch.float32)

        denom = weights.sum().item()

        if denom <= 0:
            score = window_ent_means.mean().item()
        else:
            score = (weights * window_ent_means).sum().item() / denom

        method_records.append((answer_text, score, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in rank_weighted_confidence")

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