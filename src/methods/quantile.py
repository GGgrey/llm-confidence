import torch.nn.functional as F
import torch

from src.utils import aggregate_paths_based_on_scores


def quantile(sample_paths, method_cfg, config):
    
    method_records = []
    alpha = method_cfg["alpha"]
    confidence_method = method_cfg["confidence"]

    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        probs = F.softmax(output_scores, dim=-1)

        if confidence_method == "prob":
            token_probs = probs.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1)
            s_quant = torch.quantile(token_probs, alpha).item()
        elif confidence_method == "entropy":
            eps = 1e-12
            token_entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=torch.float))
            token_certainty = max_entropy - token_entropy
            norm_certainty = token_certainty / max_entropy
            s_quant = torch.quantile(norm_certainty, alpha).item()
        elif confidence_method == "logit":
            token_logits = output_scores.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1)
            s_quant = torch.quantile(token_logits, alpha).item()
        else:
            raise ValueError(f"Unsupported confidence calculation mode: {confidence_method}")

        method_records.append((answer_text, s_quant, final_answer))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return (max(method_records, key=lambda x: x[1]))