from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import os
from typing import List, Optional, Tuple
from statistics import mean, pstdev
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmlingua import PromptCompressor


def get_available_gpus(exclude_list: str):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return []
    try:
        if not exclude_list:
            exclude = set()
        else:
            exclude = set(int(x.strip()) for x in exclude_list.split(",") if x.strip())
    except ValueError as e:
        print(f"Warning: Invalid format in exclude_gpus string ('{exclude_list}')")
        exclude = set()
    
    available = [i for i in range(num_gpus) if i not in exclude]

    if not available:
        raise RuntimeError("No GPUs available after applying exclusion list")
    
    return available


def load_jsonl(file):
    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: 
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Error parsing JSON at line {i+1} in {file}")
                continue
            except Exception as e:
                print(f"Warning: Unexpected error loading line {i+1} in {file}: {e}")
                continue
    print("Load file: ", file)


def save_jsonl(samples, save_path):
    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Save to: ", save_path)


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(model_name, read_model_from_huggingface=True):
    print(f"Loading model: {model_name}")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            local_files_only=read_model_from_huggingface,
            trust_remote_code=True,
            torch_dtype=dtype,
            # attn_implementation="eager"
        )
        model = model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer: {e}")


def load_lingua_model(lingua_model_name):
    print(f"Loading Lingua model: {lingua_model_name}")
    llm_lingua = PromptCompressor(
        model_name=lingua_model_name,
        use_llmlingua2=True,
    )
    return llm_lingua


def load_and_sample_parquet_datasets(data_dir, dataset_files, number_samples, seed):
    loaded_datasets = {}
    for filename, file_path in dataset_files.items():
        full_path  = os.path.join(data_dir, file_path)
        if os.path.isfile(full_path ):
            try:
                df = pd.read_parquet(full_path )
                df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
                if len(df) > number_samples:
                    df = df.head(number_samples)
                loaded_datasets[filename] = df
                print(f"Loaded {filename}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading parquet {filename}: {e}")
        else:
            print(f"File not found: {full_path }")

    return loaded_datasets


def load_datasets(data_dir, dataset_files):
    loaded_datasets = {}
    for filename, file_path in dataset_files.items():
        full_path = os.path.join(data_dir, file_path)
        try:
            df = pd.read_json(full_path, lines=True)  # jsonl -> DataFrame
            loaded_datasets[filename] = df
        except Exception as e:
            print(f"Load {filename} failed: {e}")
    
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

    batch_inputs = []

    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None

    for question in batch_questions:
        content = construct_prompt(
            question=question,
            use_base_prompt=use_base_prompt
        )

        if has_chat_template:
            messages = [{"role": "user", "content": content}]
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = f"User: {content}\nAssistant:"

        batch_inputs.append(input_text)
    
    tokenized_batch = tokenizer(
        batch_inputs,
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


def find_last_subsequence_token_spans(full_text, sub_text, tokenizer):
    if not sub_text:
        return None, None
    
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

    answers, spans = extract_all_numerical_values_and_spans(full_text)
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


def extract_all_numerical_values_and_spans(text):
    pattern = re.compile(r'([+-]?\d?[0-9.,/*\-+]*\d)')
    
    match = []
    span = []
    
    for m in pattern.finditer(text):
        match.append(m.group())
        span.append((m.start(), m.end()))
    
    return match, span


def aggregate_paths_based_on_scores(paths):
    if not paths:
        return None, 0.0, None

    answer_scores = {}
    for _, delta, final_answer in paths:
        answer_scores[final_answer] = answer_scores.get(final_answer, 0.0) + float(delta)

    best_final_answer = max(answer_scores, key=answer_scores.get)

    candidates = [
        (answer_text, delta, final_answer)
        for (answer_text, delta, final_answer) in paths
        if final_answer == best_final_answer
    ]
    if not candidates:
        raise RuntimeError("Best final answer not found in paths")
    
    best_answer_text, best_delta, _ = max(candidates, key=lambda t: t[1])

    return best_answer_text, best_delta, best_final_answer


def _winsorize(values, lower_q=0.05, upper_q=0.95):
    if not values:
        return []
    xs = sorted(values)
    n = len(xs)
    lo = xs[int((n - 1) * lower_q)]
    hi = xs[int((n - 1) * upper_q)]
    return [min(max(x, lo), hi) for x in values]


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def aggregate_paths_lcb_robust_vote(paths):
    grouped_scores = defaultdict(list)
    grouped_paths = defaultdict(list)
    for answer_text, score, final_answer in paths:
        grouped_scores[final_answer].append(float(score))
        grouped_paths[final_answer].append((answer_text, float(score), final_answer))
    
    answer_scores = {}
    for final_answer, scores in grouped_scores.items():
        winsorized_scores = _winsorize(scores, lower_q=0.05, upper_q=0.95)
        mean_score, std_score = _mean_std(winsorized_scores)
        n = len(winsorized_scores)
        lcb_score = mean_score - 1.96 * (std_score / math.sqrt(n + 1e-6))
        answer_scores[final_answer] = lcb_score * math.log(n + 1)

    best_final = max(answer_scores, key=answer_scores.get)
    best_answer_text = max(grouped_paths[best_final], key=lambda x: x[1])[0]

    return best_answer_text, answer_scores[best_final], best_final


def aggregate_paths_based_on_scores_using_min(paths):

    if not paths:
        return None, 0.0, None
    
    y_max = max(y for _, y, _ in paths)
    normalized_paths = [(x, y_max - y, z) for x, y, z in paths]  # Larger is better

    answer_scores = {}
    for _, delta, final_answer in normalized_paths:
        answer_scores[final_answer] = answer_scores.get(final_answer, 0.0) + float(delta)

    best_final_answer = max(answer_scores, key=answer_scores.get)

    candidates = [
        (answer_text, y, final_answer)
        for (answer_text, y, final_answer) in paths
        if final_answer == best_final_answer
    ]
    if not candidates:
        raise RuntimeError("Best final answer not found in original paths")
    
    best_answer_text, best_y, _ = min(candidates, key=lambda t: t[1])

    return best_answer_text, best_y, best_final_answer


def construct_p_true_prompt(question, answer):
    base_prompt = f"Please answer either with 'True' or 'False' only. Is it True that: {question} {answer}"
    return base_prompt


def save_results_to_csv(results, filename):
    try:
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False)
    except Exception as e:
        print(f"Failed to save CSV to {filename}: {e}")


def print_final_accuracy_table(method_final_acc):
    print("\n=== Final Accuracies ===")
    for method_name, acc in method_final_acc.items():
        print(f"{method_name}: {acc:.2f}%")


def _json_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if "torch" in str(type(obj)):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)

    return str(obj)


def save_results_to_json(payload, file_path: str) -> Path:
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)

    return file_path


if __name__ == "__main__":
    exclude = "0, 1, 2"
    gpu_list = get_available_gpus(exclude)
    print(gpu_list)