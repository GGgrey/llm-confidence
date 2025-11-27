import torch.nn.functional as F

from src.utils.utils import aggregate_paths_based_on_scores_using_min


def likelihood(sample_paths, normalized_length, tokenizer, config):
    
    method_records = []

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        log_probs = F.log_softmax(output_scores, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1)

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            token_log_probs = token_log_probs[mask]

        if normalized_length:
            ll = - (token_log_probs).sum() / len(token_log_probs)
        else:
            ll = - (token_log_probs).sum()

        method_records.append((answer_text, ll.item(), final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores_using_min(method_records)
    else:
        return (min(method_records, key=lambda x: x[1]))