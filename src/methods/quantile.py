import torch.nn.functional as F
import torch

from src.utils import aggregate_paths_based_on_scores


def quantile(sample_paths, method_cfg, config):
    
    method_records = []
    alpha = method_cfg["alpha"]

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        probs = F.softmax(output_scores, dim=-1)

        token_probs = probs.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1)

        s_quant = torch.quantile(token_probs, alpha).item()

        method_records.append((answer_text, s_quant, final_answer))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return (max(method_records, key=lambda x: x[1]))