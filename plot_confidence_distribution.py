import random
import os
from typing import Optional

from tqdm import tqdm
import argparse
import multiprocessing as mp
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from dynasor.core.evaluator import math_equal
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import ks_2samp


def get_available_gpus(exclude_list: str):
    num_gpus = torch.cuda.device_count()
    exclude = set([int(x) for x in exclude_list.split(",") if x.strip() != ""])
    available = [i for i in range(num_gpus) if i not in exclude]
    if not available:
        raise RuntimeError("No GPUs available after applying exclusion list.")
    return available


def load_dataset(file_path):
    df = None
    if os.path.isfile(file_path):
        df = pd.read_parquet(file_path)
    return df


def construct_prompt(question, tokenizer):
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    system_prompt = system_prompt.replace("\n", "")
    content = question
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    tokenized_message = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    return tokenized_message


def extract_answer(text: str) -> Optional[str]:
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return None
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


def compute_average_trace_confidence(output_scores, k=5):
    logits = torch.stack(output_scores, dim=0)
    probs = F.softmax(logits, dim=-1)
    topk_probs, _ = torch.topk(probs, k=k, dim=-1)
    topk_probs = torch.clamp(topk_probs, min=1e-12)
    token_confidences = -torch.mean(torch.log(topk_probs), dim=-1)
    avg_conf = torch.mean(token_confidences).item()
    return avg_conf


def compute_logit_confidence(output_scores, answer_ids):
    trace_logits = []
    for token_idx, logits in enumerate(output_scores):
        token_id = answer_ids[token_idx].item()
        logit_value = logits[token_id].item()
        trace_logits.append(logit_value)
    avg_logit = float(np.mean(trace_logits)) if trace_logits else 0.0
    return avg_logit


def compute_likelihood_confidence(output_scores, answer_ids):
    trace_logps = []
    for token_idx, logits in enumerate(output_scores):
        logits = logits.view(-1)
        token_id = answer_ids[token_idx].item()
        probs = torch.softmax(logits, dim=-1)
        token_prob = probs[token_id].item()
        token_prob = max(token_prob, 1e-12)
        trace_logps.append(np.log(token_prob))
    avg_likelihood = float(np.mean(trace_logps)) if trace_logps else 0.0
    return avg_likelihood


def trend_linear_regression(entropy):
    x = np.arange(len(entropy)).reshape(-1, 1)
    y = np.array(entropy)
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(x, y)
    trend = "upward" if slope > 0 else "downward"
    return slope, intercept, r2, trend


def compute_entropy_trend_confidence(output_scores, answer_ids, tokenizer):
    logits = torch.stack(output_scores, dim=0)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    if tokenizer.pad_token_id is not None:
        mask = answer_ids != tokenizer.pad_token_id
        entropy = entropy[mask]
    slope, _, _, _ = trend_linear_regression(entropy.detach().cpu().numpy())
    return slope


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


def run_inference(
    model_name,
    prompt_chunk,
    answer_chunk,
    worker_id,
    available_gpus,
    confidence_method,
    temperature=0.1,
    max_new_tokens=2048,
):
    if len(available_gpus) == 0:
        raise RuntimeError("No CUDA devices available.")
    
    selected_gpu = available_gpus[worker_id % len(available_gpus)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    print(f"Worker {worker_id} is using GPU {selected_gpu}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    outputs = []
    for i, (prompt, ground_truth) in tqdm(
        enumerate(zip(prompt_chunk, answer_chunk)),
        total=len(prompt_chunk),
        desc=f"Worker {worker_id} (process {os.getpid()})",
        position=worker_id,
        leave=True,
    ):
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = input_ids["input_ids"].shape[-1]

        output = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        generated_ids = output.sequences[0]
        answer_ids = generated_ids[prompt_len:]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        output_scores = [s[0].to("cpu") for s in output.scores]

        # Extract final answer string
        final_answer = extract_answer(answer_text)

        # Calculate confidence
        confidence = 0.0
        if confidence_method == "mean_confidence":
            confidence = compute_average_trace_confidence(output_scores, k=20)
        elif confidence_method == "logit":
            confidence = compute_logit_confidence(output_scores, answer_ids)
        elif confidence_method == "likelihood":
            confidence = compute_likelihood_confidence(output_scores, answer_ids)
        elif confidence_method == "entropy_trend":
            confidence = compute_entropy_trend_confidence(output_scores, answer_ids, tokenizer)
        else:
            raise ValueError("Confidence method is not supported.")

        is_valid = final_answer is not None
        try:
            is_correct = equal_func(final_answer, ground_truth)
        except:
            is_correct = str(final_answer) == str(ground_truth)

        outputs.append({
            "text": answer_text,
            "answer": final_answer,
            "ground_truth": ground_truth,
            "confidence": confidence,
            "is_valid": is_valid,
            "is_correct": is_correct
        })

        del output, input_ids
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    
    return outputs


def run(args):

    available_gpus = get_available_gpus(args.exclude_gpus)

    dataset = load_dataset(args.data_dir)

    questions = dataset["question"].tolist()
    questions = questions[:500]
    answers = dataset["numeric_final_answer"].astype(str).tolist()
    answers = answers[:500]
    completions = []

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.confidence_method == "mean_confidence" or \
        args.confidence_method == "logit" or \
        args.confidence_method == "likelihood" or \
        args.confidence_method == "entropy_trend":

        mp.set_start_method("spawn")
        num_processes = args.num_processes

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        tokenized_massages = [
            construct_prompt(question, tokenizer)
            for question in questions
        ]

        upsampled_tokenized_massages = tokenized_massages * args.num_samples
        upsampled_answers = answers * args.num_samples

        k, m = divmod(len(upsampled_tokenized_massages), num_processes)
        prompt_chunks = [
            upsampled_tokenized_massages[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(num_processes)
        ]
        answer_chunks = [
            upsampled_answers[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(num_processes)
        ]

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                run_inference,
                [
                    (
                        args.model,
                        prompt_chunks[i],
                        answer_chunks[i],
                        i,
                        available_gpus,
                        args.confidence_method,
                        args.temperature,
                        args.max_new_tokens
                    )
                    for i in range(num_processes)
                ],
            )
        
        completions = [output for sublist in results for output in sublist]
    
    if not completions or len(completions) != len(upsampled_tokenized_massages):
        raise ValueError("Completions are not valid.")

    df = pd.DataFrame(completions)
    df = df[df["is_valid"]]
    correct_conf = df[df["is_correct"]]["confidence"]
    incorrect_conf = df[~df["is_correct"]]["confidence"]

    print(f"Valid count: {len(df)}")
    print(f"Correct count: {len(correct_conf)}, Incorrect count: {len(incorrect_conf)}")
    print(f"Average accuracy: {df['is_correct'].mean():.2%}")

    ks_stat, p_value = ks_2samp(correct_conf, incorrect_conf)
    print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4e}")

    plt.figure(figsize=(8, 4))
    bins = np.linspace(df["confidence"].min(), df["confidence"].max(), 80)

    plt.hist(correct_conf, bins=bins, alpha=0.7, color="seagreen", label="Correct")
    plt.hist(incorrect_conf, bins=bins, alpha=0.7, color="peru", label="Incorrect")

    plt.axvline(correct_conf.mean(), color="darkgreen", ls="--", lw=1.5, label="Correct Mean")
    plt.axvline(incorrect_conf.mean(), color="saddlebrown", ls="--", lw=1.5, label="Incorrect Mean")

    plt.xlabel("Confidence", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    title_str = (
        f"Method: {args.confidence_method} | "
        f"KS stat: {ks_stat:.4f}, p: {p_value:.2e}"
    )
    plt.title(title_str, fontsize=14)
    plt.legend()
    plt.tight_layout()

    base_dir = os.path.join("outputs/confidence_distribution/", f"{args.confidence_method}")
    os.makedirs(base_dir, exist_ok=True)
    fig_save_path = os.path.join(
        base_dir,
        f"{args.confidence_method}_plot.png"
    )
    plt.savefig(fig_save_path, dpi=300)
    plt.close()

    csv_save_path = os.path.join(
        base_dir,
        "inference_results.csv"
    )
    df.to_csv(csv_save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", "-d", type=str, default="data/src_custom_datasets_math_dataset_test_processed.parquet")
    parser.add_argument("--model", type=str, default="/data/sunqiao/projects/models/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--confidence_method",
        type=str,
        default="entropy_trend",
        choices=[
            "mean_confidence",
            "perplexity",
            "likelihood",
            "logit",
            "entropy_trend"
        ],
    )
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=float, default=2048)
    parser.add_argument("--num_processes", default=4, type=int)
    parser.add_argument(
        "--exclude_gpus",
        type=str,
        default="",
        help="Comma-separated list of GPU ids to exclude, e.g. '2' or '1,3'."
    )

    args = parser.parse_args()

    run(args)