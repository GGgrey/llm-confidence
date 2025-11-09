import random
import re
import os
from typing import List, Optional, Tuple
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmlingua import PromptCompressor
from dynasor.core.evaluator import math_equal


def seed_everything(seed=0):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_model_and_tokenizer(model_name, read_model_from_huggingface=True):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        local_files_only=read_model_from_huggingface,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def load_lingua_model(lingua_model_name):

    llm_lingua = PromptCompressor(
        model_name=lingua_model_name,
        use_llmlingua2=True,
    )

    return llm_lingua


def load_and_sample_parquet_datasets(data_dir, dataset_files, number_samples, seed):

    loaded_datasets = {}
    for filename, file_path in dataset_files.items():
        file_path = os.path.join(data_dir, file_path)
        if os.path.isfile(file_path):
            df = pd.read_parquet(file_path)
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            if len(df) > number_samples:
                df = df.head(number_samples)
            loaded_datasets[filename] = df

    return loaded_datasets


def construct_prompt(question, use_base_prompt=False):

    if use_base_prompt:
        base_prompt = (
            f"Question: {question}\n"
            f"Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
            f"Note that the answer must be enclosed in the \\boxed{{X}}, where X is the final answer, at the end of your response."
        )
    else:
        base_prompt = (
            f"Question: {question}\n"
            f"Provide your step-by-step reasoning first and express the answer at the end of each step in the format \"The answer is \\boxed{{A}}\", where A is the answer of the step.\n"
            f"After completing all the steps, print \"The final answer is \\boxed{{X}}\", "
            f"Note that the answer must be enclosed in the \\boxed{{X}}, where X is the final answer, at the end of your response."
        )
    
    return base_prompt


def setup_tokenizer_padding_config(tokenizer, model):

    if tokenizer.pad_token is None:
        if "llama" in tokenizer.name_or_path.lower():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"


def batch_messages_creation(tokenizer, batch_questions, device, use_base_prompt):

    batch_messages = []
    for question in batch_questions:
        batch_messages.append([
            {
                "role": "user",
                "content": construct_prompt(
                    question=question,
                    use_base_prompt=use_base_prompt
                )
            }
        ])
    
    batch_template_messages = []
    for message in batch_messages:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            input_text = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in message])
            input_text += "\nassistant:"
        
        batch_template_messages.append(input_text)
    
    tokenized_batch = tokenizer(
        batch_template_messages,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    tokenized_batch = tokenized_batch.to(device)

    return tokenized_batch


def extract_answer(text: str) -> Optional[str]:
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    
    return None


def extract_all_boxed_answer_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    answers = []
    spans = []
    i = 0
    while i < len(text):
        if text[i:i+6] == "\\boxed":
            i += 6
            if i < len(text) and text[i] == "{":
                i += 1
                start = i
                stack = 1
                answer_chars = []
                while i < len(text) and stack > 0:
                    if text[i] == "{":
                        stack += 1
                        answer_chars.append(text[i])
                    elif text[i] == "}":
                        stack -= 1
                        if stack > 0:
                            answer_chars.append(text[i])
                    else:
                        answer_chars.append(text[i])
                    i += 1
                end = i - 1
                answers.append("".join(answer_chars).strip())
                spans.append((start, end))
            else:
                start = i
                answer_chars = []
                while i < len(text) and text[i] not in [" ", "\n"]:
                    answer_chars.append(text[i])
                    i += 1
                end = i - 1
                answers.append("".join(answer_chars).strip())
                spans.append((start, end))
        else:
            i += 1
    return answers, spans


def quick_parse(text: str) -> str:
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def find_last_subsequence_token_spans(full_text, sub_text, tokenizer):
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


def find_all_subsequence_token_spans(full_text, tokenizer):
    token_spans = []

    full_text_tokenized = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = full_text_tokenized['offset_mapping']

    answers, spans = extract_all_boxed_answer_spans(full_text)
    if not answers or not spans or len(answers) != len(spans):
        return token_spans
    
    for answer, (answer_text_start_idx, answer_text_end_idx) in zip(answers, spans):
        answer_token_start_idx = None
        answer_token_end_idx = None
        
        for i, (s, e) in enumerate(offsets):
            if s <= answer_text_start_idx < e:
                answer_token_start_idx = i
            if s < answer_text_start_idx + len(answer) <= e:
                answer_token_end_idx = i + 1
            if answer_token_start_idx is not None and answer_token_end_idx is not None:
                break
        token_spans.append((answer_token_start_idx, answer_token_end_idx))
    
    return token_spans


def postprocess_final_answer(numeric_expression: str) -> str:
    try:
        cleaned_up = numeric_expression.replace(',', '')
        result = eval(cleaned_up)
        return str(result)
    except Exception:
        print(f'Can not process {numeric_expression}')
        return numeric_expression
    

def extract_last_numerical_value(text):
    matches = re.findall(r'([+-]?\d?[0-9.,/*\-+]*\d)', text)
    return matches[-1] if matches else None


def extract_proper_nouns(doc):
    proper_nouns = []
    current_proper_noun = []

    for token in doc:
        if token.pos_ == "PROPN":
            current_proper_noun.append(token.text)
        elif current_proper_noun and (token.like_num or "-" in token.text):
            current_proper_noun.append(token.text)
        elif current_proper_noun:
            proper_nouns.append(" ".join(current_proper_noun))
            current_proper_noun = []

    if current_proper_noun:
        proper_nouns.append(" ".join(current_proper_noun))

    return proper_nouns


def extract_final_answer(text):
    pattern = r"The final answer is ([^,\.]+)"
    match = re.search(pattern, text)
    return match.group(1) if match else None


def find_subsequence_indices(sequence, subsequence, occurrence_count: int = 1):
    found_count = 0
    seq_len = len(sequence)
    sub_len = len(subsequence)

    for i in range(seq_len - sub_len + 1):
        if sequence[i:i + sub_len] == subsequence:
            if found_count == occurrence_count - 1:
                return i
            found_count += 1
    return -1


def extract_all_steps(text):
    regex = r"(Step \d+:.*?)(?=\n\n|$)"
    return re.findall(regex, text, re.DOTALL)


def extract_all_numerical_values(text):
    return re.findall(r'([+-]?\d?[0-9.,/*\-+]*\d)', text)


def aggregate_paths_based_on_scores(paths):
    answer_scores = {}
    answer_text = None
    for answer, delta, final_answer in paths:
        answer_scores[final_answer] = answer_scores.get(final_answer, 0) + delta

    best_answer = max(answer_scores, key=answer_scores.get)

    for answer, delta, final_answer in paths:
        if final_answer == best_answer:
            answer_text = answer
            break

    return answer_text, answer_scores[best_answer], best_answer


def construct_p_true_prompt(question, answer):
    base_prompt = f"Please answer either with 'True' or 'False' only. Is it True that: {question} {answer}"
    return base_prompt


def aggregate_paths_based_on_scores_using_min(paths):
    
    y_max = max(y for _, y, _ in paths)
    normalized_paths = [(x, y_max - y, z) for x, y, z in paths]

    answer_scores = {}
    answer_text = None
    for answer, delta, final_answer in normalized_paths:
        answer_scores[final_answer] = answer_scores.get(final_answer, 0) + delta

    best_answer = max(answer_scores, key=answer_scores.get)

    for answer, delta, final_answer in normalized_paths:
        if final_answer == best_answer:
            answer_text = answer
            break

    return answer_text, answer_scores[best_answer], best_answer


def save_results_to_csv(results, filename):
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)


def print_final_accuracy_table(method_final_acc):
    print("\n=== Final Accuracies ===")
    for method_name, acc in method_final_acc.items():
        print(f"{method_name}: {acc:.2f}%")