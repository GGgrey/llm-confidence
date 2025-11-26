from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np
import torch

from src.utils import aggregate_paths_based_on_scores, extract_last_numerical_value


def plot_token_importance(token_importance, tokens, tokenizer):
    
    token_texts = tokenizer.convert_ids_to_tokens(tokens)

    plt.figure(figsize=(10, 4))

    plt.plot(range(len(token_importance)), token_importance, marker='o', linestyle='-', markersize=3, linewidth=0.8, alpha=0.7)
    plt.xticks(range(len(token_texts)), token_texts, rotation=45, fontsize=4)
    plt.xlabel("Tokens")
    plt.ylabel("Importance Score")
    plt.title("Token Importance Curve")
    plt.grid(True)

    for i, score in enumerate(token_importance):
        plt.text(i, score, f"{score:.2f}", ha='center', va='bottom', fontsize=4)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"token_importance_{timestamp}.png"
    plt.savefig(filename, dpi=300)

    plt.show()


def get_attn_weights(generated_ids, model):

    attn_weights = None

    with torch.no_grad():
        outputs = model(
            generated_ids.unsqueeze(0),
            output_attentions=True,
            return_dict=True
        )

        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attn_weights = outputs.attentions  # tuple: (num_layers, batch, num_heads, seq, seq)
    
    if attn_weights is None:
        return None
    
    return attn_weights


def find_last_subsequence_token_spans(full_text, sub_text, tokenizer):
    """Find the token spans of last occurrence of a subsequence in a full text"""
    final_answer_token_start_idx = None
    final_answer_token_end_idx = None

    full_text_tokenized = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = full_text_tokenized['offset_mapping']

    final_answer_start_idx = full_text.rfind(sub_text)
    if final_answer_start_idx == -1:
        return None, None
    
    for i, (s, e) in enumerate(offsets):
        if s <= final_answer_start_idx < e:
            final_answer_token_start_idx = i
        if s < final_answer_start_idx + len(sub_text) <= e:
            final_answer_token_end_idx = i + 1
        if final_answer_token_start_idx is not None and final_answer_token_end_idx is not None:
            break
    
    return final_answer_token_start_idx, final_answer_token_end_idx


def attention_weighted_confidence(sample_paths, tokenizer, model, config):
    
    method_records = []

    for path in sample_paths:
        generated_ids = path["generated_ids"]
        final_answer = path["final_answer"]
        answer_ids = path["answer_ids"]
        answer_text = path["answer_text"]
        output_scores = path["output_scores"]
        prompt_len = path["prompt_len"]

        if tokenizer.pad_token_id is not None:
            mask = generated_ids != tokenizer.pad_token_id
            generated_ids = generated_ids[mask]

            mask = answer_ids != tokenizer.pad_token_id
            answer_ids = answer_ids[mask]
            output_scores = output_scores[mask]

            prompt_len = len(generated_ids) - len(answer_ids)

        final_answer_ori = extract_last_numerical_value(answer_text)

        final_answer_token_start_idx, final_answer_token_end_idx = find_last_subsequence_token_spans(
            answer_text, final_answer_ori, tokenizer)
        if (
            final_answer_token_start_idx is None
            or final_answer_token_end_idx is None
            ):
            print(f"Final answer is not in the answer text, final answer: {final_answer}, answer text: {answer_text}")
            method_records.append((answer_text, 0.0, final_answer))
            continue

        attn_weights = get_attn_weights(generated_ids, model)

        num_layers = len(attn_weights)
        num_heads = attn_weights[0].shape[1]

        token_importance = np.zeros(final_answer_token_start_idx)

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                full_attn = attn_weights[layer_idx][0, head_idx].to(torch.float).cpu().numpy()  # (seq, seq)
                answer_text_attn = full_attn[prompt_len:, prompt_len:]  # (answer_len, answer_len)
                final_answer_attn = answer_text_attn[final_answer_token_start_idx:final_answer_token_end_idx, :]

                for token_idx in range(len(token_importance)):
                    token_importance[token_idx] += final_answer_attn[:, token_idx].sum().item()

        token_importance /= (num_layers * num_heads)

        token_importance = np.log1p(token_importance)
        low, high = np.percentile(token_importance, [5, 95])
        token_importance = np.clip(token_importance, low, high)
        token_importance = (token_importance - token_importance.min()) / (token_importance.max() - token_importance.min())

        # plot_token_importance(token_importance, answer_ids[:final_answer_token_start_idx], tokenizer)

        confidence = 0.0

        for token_idx in range(len(token_importance)):
            token_logits = output_scores[token_idx]
            probs = torch.softmax(token_logits, dim=-1)
            token_prob = probs[answer_ids[token_idx]]

            token_score = (token_prob.item() * token_importance[token_idx])
 
            confidence += token_score

        confidence = confidence / len(token_importance)

        method_records.append((answer_text, confidence, final_answer))
    
    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")

    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return (max(method_records, key=lambda x: x[1]))





        


