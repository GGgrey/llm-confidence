import re

import numpy as np
import torch

from src.utils.utils import aggregate_paths_based_on_scores, find_last_subsequence_token_spans, find_all_subsequence_token_spans
from src.utils.parser import _last_boxed


def calculate_confidence_for_all_answers(logits_spans, answer_ids, token_spans, confidence_method):
    confidence_list = []

    for logits, (answer_token_start_idx, answer_token_end_idx) in zip(logits_spans, token_spans):

        if logits is None:
            confidence_list.append(0.0)
            continue

        if answer_token_end_idx < 0 or answer_token_end_idx >= len(answer_ids) or answer_token_start_idx >= answer_token_end_idx:
            confidence_list.append(0.0)
            continue

        answer_token_ids = answer_ids[answer_token_start_idx:answer_token_end_idx]

        if confidence_method == "product":
            confidence_all = 1.0
        elif confidence_method == "sum":
            confidence_all = 0.0
        elif confidence_method == "entropy":
            confidence_all = 0.0
        else:
            raise ValueError(f"Unsupported confidence calculation mode: {confidence_method}")
        valid_tokens = 0

        for i, token_id in enumerate(answer_token_ids):

            token_logits = logits[i]

            probs = torch.softmax(token_logits, dim=-1)
            token_prob = probs[token_id]

            if confidence_method == "product":
                confidence_all *= token_prob.item()
            elif confidence_method == "sum":
                confidence_all += token_prob.item()
            elif confidence_method == "entropy":
                probs = torch.clamp(probs, min=1e-12)
                entropy = - (probs * torch.log(probs)).sum(dim=-1)
                confidence_all += (1.0 - entropy.item() / np.log(len(probs)).item())
            else:
                raise ValueError(f"Unsupported confidence calculation mode: {confidence_method}")
            
            valid_tokens += 1
        
        confidence = confidence_all / valid_tokens if valid_tokens > 0 else 0.0
        confidence_list.append(confidence)

    return confidence_list 


def calculate_confidence_for_final_answer(
    logits,
    answer_ids,
    confidence_method="product"
):
    if confidence_method == "product":
        confidence_all = 1.0
    elif confidence_method == "sum":
        confidence_all = 0.0
    elif confidence_method == "entropy":
        confidence_all = 0.0
    else:
        raise ValueError(f"Unsupported confidence calculation mode: {confidence_method}")
    valid_tokens = 0

    for idx, token_id in enumerate(answer_ids):

        if idx >= len(logits):
            break

        token_logits = logits[idx]

        probs = torch.softmax(token_logits, dim=-1)
        token_prob = probs[token_id]

        if confidence_method == "product":
            confidence_all *= token_prob.item()
        elif confidence_method == "sum":
            confidence_all += token_prob.item()
        elif confidence_method == "entropy":
            probs = torch.clamp(probs, min=1e-12)
            entropy = - (probs * torch.log(probs)).sum(dim=-1)
            confidence_all += (1.0 - entropy.item() / np.log(len(probs)).item())
        else:
            raise NotImplementedError("Unsupported confidence calculation mode")
        
        valid_tokens += 1
    
    return confidence_all / valid_tokens if valid_tokens > 0 else 0.0


def handle_last_decoding(
    tokenizer,
    answer_text,
    final_answer,
    output_scores,
    answer_ids,
    confidence_method,
):
    confidence = 0.0

    if any(v is None for v in (answer_ids, answer_text, final_answer, output_scores)):
        print("Invalid trace")
        return confidence
    
    extracted_answer = _last_boxed(answer_text)
    if not extracted_answer:
        print("Can not extract answer")
        return confidence
    
    final_answer_token_start_idx, final_answer_token_end_idx = find_last_subsequence_token_spans(
        answer_text,
        extracted_answer,
        tokenizer
    )
    
    if final_answer_token_start_idx is None or final_answer_token_end_idx is None:
        print("Can not find final answer index")
        return confidence
    
    if final_answer_token_start_idx < 0 or final_answer_token_end_idx > len(answer_ids) - 1:
        print("Final answer index is out of range")
        return confidence
    
    decoded_answer = tokenizer.decode(answer_ids[final_answer_token_start_idx: final_answer_token_end_idx])
    print(f"Decoded answer: {decoded_answer}")
    
    final_answer_scores = output_scores[
        final_answer_token_start_idx: final_answer_token_end_idx
    ]

    confidence = calculate_confidence_for_final_answer(
        final_answer_scores,
        answer_ids[final_answer_token_start_idx: final_answer_token_end_idx],
        confidence_method
    )
    
    return confidence


def get_logits_span(logits, token_spans):
    """Get the logits for each token span"""
    logits_spans = []

    for (start, end) in token_spans:
        if start < 0 or end > len(logits) or start >= end:
            logits_spans.append(None)
        else:
            logits_spans.append(logits[start:end])

    return logits_spans


def handle_all_decoding(
    tokenizer,
    answer_text,
    final_answer,
    output_scores,
    answer_ids,
    scoring_mode,
    confidence_method,
):
    confidence = 0.0
    valid_count = 0

    if any(v is None for v in (answer_ids, answer_text, final_answer, output_scores)):
        return confidence

    token_spans = [
        s
        for s in find_all_subsequence_token_spans(answer_text, tokenizer)
        if s is not None
        and len(s) == 2
        and None not in s
    ]

    if not token_spans:
        return confidence
    
    logits_spans = get_logits_span(output_scores, token_spans)
    if not logits_spans:
        return confidence
    
    confidence_list = calculate_confidence_for_all_answers(logits_spans,
                                                           answer_ids,
                                                           token_spans,
                                                           confidence_method)
    
    valid_count = len(confidence_list)
    if not confidence_list or valid_count == 0:
        return confidence

    if scoring_mode == "log":  # (lop(1 + c1) + ... + log(1 + cn)) / n
        confidence = np.sum(np.log(1 + np.array(confidence_list, dtype=float)))
        confidence = confidence / valid_count

    elif scoring_mode == "min":  # min(c1, ..., cn)
        confidence = np.min(confidence_list)

    elif scoring_mode == "max":  # max(c1, ..., cn)
        confidence = np.max(confidence_list)

    elif scoring_mode == "h_mean":  # n / (1/c1 + ... + 1/cn)
        confidence = np.sum(1 / (1e-11 + np.array(confidence_list, dtype=float)))
        confidence = valid_count / confidence if confidence > 0 else 0.0

    elif scoring_mode == "mean":  # (c1 + ... + cn) / n
        confidence = np.mean(confidence_list)

    elif scoring_mode == "weighted_mean":  # (1*c1 + ... + n*cn) / (1 + ... + n)
        confidence = np.sum(np.array([i * c for i, c in enumerate(confidence_list, start=1)], dtype=float))
        confidence = confidence / (valid_count * (valid_count + 1) / 2)

    elif scoring_mode == "weighted_half":  # half: final answer, half: all the others
        if valid_count == 1:
            confidence = confidence_list[0]
        else:
            confidence = (0.5 * confidence_list[-1] + 0.5 * np.sum(confidence_list[:-1]) / (valid_count - 1))
    
    elif scoring_mode == "weighted_2":  # (2^0*c1 + ... + 2^(n-1)*cn) / (2^n - 1)
        try:
            weights = [2**i for i in range(valid_count)]
            confidence = np.sum(np.array([w * c for w, c in zip(weights, confidence_list)], dtype=float))
            confidence = confidence / (2**valid_count - 1)
        except OverflowError:
            threshold = 100  # Avoid overflow in weights, keep only the last 100 elements
            trimmed_confidence_list = confidence_list[-threshold:]
            weights = [2**i for i in range(len(trimmed_confidence_list))]
            total_weight = 2**len(trimmed_confidence_list) - 1
            confidence = np.sum(np.array([w * c for w, c in zip(weights, trimmed_confidence_list)], dtype=float))
            confidence = confidence / total_weight

    else:
        raise ValueError(f"Unsupported scoring mode: {scoring_mode}")
    
    return confidence


def cer(sample_paths, method_cfg, tokenizer, config):

    method_records = []
    decoding_mode = method_cfg["decoding_mode"]
    scoring_mode = method_cfg["scoring_mode"]
    confidence_method = method_cfg["confidence"]

    for idx, path in enumerate(sample_paths):
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]

        if tokenizer.pad_token_id is not None:
            mask = answer_ids != tokenizer.pad_token_id
            answer_ids = answer_ids[mask]
            output_scores = output_scores[mask]

        if decoding_mode == "last":
            confidence = handle_last_decoding(tokenizer, answer_text, final_answer, 
                                              output_scores, answer_ids,
                                              confidence_method)
        elif decoding_mode == "all":
            confidence = handle_all_decoding(tokenizer, answer_text, final_answer, 
                                             output_scores, answer_ids,
                                             scoring_mode, confidence_method)
        else:
            raise ValueError(f"Unsupported decoding mode: {decoding_mode}")
        
        method_records.append((answer_text, confidence, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Error happened in cer")
    
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


