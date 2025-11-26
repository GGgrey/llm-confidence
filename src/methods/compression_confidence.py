import torch
import torch.nn.functional as F

from src.utils import aggregate_paths_based_on_scores


def get_output_scores(generated_ids, model):
    output_scores = None
    
    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            output_scores=True,
            return_dict=True
        )

        if hasattr(outputs, 'logits') and outputs.logits is not None:
            output_scores = outputs.logits[0]  # (seq_len, vocab_size)
    
    if output_scores is None:
        return None

    return output_scores


def compute_compression_confidence(compressed_answer_ids, prompt_len, compressed_scores, confidence_method):
    confidence = 0.0
    answer_logits = compressed_scores[prompt_len: prompt_len + len(compressed_answer_ids)]

    probs = F.softmax(answer_logits, dim=-1)
    token_probs = probs[torch.arange(len(compressed_answer_ids)), compressed_answer_ids]

    if confidence_method == "default":
        confidence = token_probs.prod().item()
    elif confidence_method == "mean":
        confidence = token_probs.mean().item()
    elif confidence_method == "min":
        confidence = token_probs.min().item()
    elif confidence_method == "max":
        confidence = token_probs.max().item()
    elif confidence_method == "median":
        confidence = token_probs.median().item()
    elif confidence_method == "geometric_mean":
        confidence = token_probs.prod().pow(1.0 / len(token_probs)).item()
    elif confidence_method == "entropy":
        safe_probs = torch.clamp(probs, min=1e-12)
        token_entropies = - (safe_probs * torch.log(safe_probs)).sum(dim=-1)
        vocab_size = safe_probs.size(-1)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=safe_probs.dtype, device=safe_probs.device))
        norm_entropies = token_entropies / max_entropy
        confidence = (1.0 - norm_entropies).mean().item()

    return confidence


def compression_confidence(sample_paths, method_cfg, model, lingua_model, tokenizer, device, config):
    
    method_records = []
    confidence_method = method_cfg["confidence"]
    compression_ratio = method_cfg["compression_ratio"]

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

        compressed_output = lingua_model.compress_prompt(answer_text, rate=compression_ratio, force_reserve_digit=True, drop_consecutive=True)
        compressed_answer_text = compressed_output['compressed_prompt']
        compressed_answer_ids = tokenizer.encode(compressed_answer_text, return_tensors='pt').squeeze(0).to(device)
        compressed_ids = torch.cat((generated_ids[:prompt_len], compressed_answer_ids), dim=0)

        compressed_scores = get_output_scores(compressed_ids, model)

        confidence = compute_compression_confidence(compressed_answer_ids, prompt_len, compressed_scores, confidence_method)

        method_records.append((answer_text, confidence, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")

    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return (max(method_records, key=lambda x: x[1]))
