import torch
import torch.nn.functional as F

from src.utils.utils import construct_p_true_prompt, aggregate_paths_based_on_scores


def p_true(sample_paths, tokenizer, model, device, config):

    method_records = []
    
    for path in sample_paths:
        final_answer = path["final_answer"]
        answer_text = path["answer_text"]
        question = path["prompt"]

        p_true_message = [{"role": "user", "content": construct_p_true_prompt(question=question, answer=answer_text)}]

        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            p_true_prompt = tokenizer.apply_chat_template(
                p_true_message,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            p_true_prompt = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in p_true_message])
            p_true_prompt += "\nassistant:"

        p_true_prompt_tokenized = tokenizer(
            p_true_prompt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        p_true_prompt_tokenized = p_true_prompt_tokenized.to(device)
        input_ids = p_true_prompt_tokenized["input_ids"]
        attention_mask = p_true_prompt_tokenized["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]

            true_token_id = tokenizer.convert_tokens_to_ids("True")
            probs = F.softmax(next_token_logits, dim=-1)
            true_token_probs = probs[:, true_token_id].item()

            method_records.append((answer_text, true_token_probs, final_answer))

    if not method_records or len(method_records) != len(sample_paths):
        raise RuntimeError("Decoding error")
    
    if config.aggregate:
        return aggregate_paths_based_on_scores(method_records)
    else:
        return (max(method_records, key=lambda x: x[1]))
