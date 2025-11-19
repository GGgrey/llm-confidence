import gc
import os
import random
import argparse
import multiprocessing as mp
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ks_2samp

from src.grader import check_is_correct
from src.utils import extract_answer


def get_available_gpus(exclude_list: str):
    num_gpus = torch.cuda.device_count()
    exclude = set(
        [int(x) for x in exclude_list.split(",") if x.strip() != ""]
    )
    available = [i for i in range(num_gpus) if i not in exclude]
    if not available:
        raise RuntimeError("No GPUs available after applying exclusion list")
    return available


def load_dataset(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    return pd.read_parquet(file_path)


def construct_prompt(question, tokenizer):
    base_prompt = (
        f"Question: {question}\n"
        f"Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
        f"Note that the answer must be enclosed in the \\boxed{{X}}, where X is the final answer, at the end of your response."
    )
    message = [
        {"role": "user", "content": base_prompt}
    ]
    tokenized_message = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    return tokenized_message
    

def compute_input_perplexity(model, tokenizer, prompt: str):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    ppl_value = torch.exp(loss).item()
    return ppl_value


def compute_attention_dispersion(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[-1].mean(dim=1) 
    attn_entropy = -(attn * torch.log(attn + 1e-12)).sum(dim=-1).mean().item()
    return -attn_entropy


def run_single_problem(model, tokenizer, question, ground_truth, num_samples, temperature, max_new_tokens):
    answers = []
    for _ in range(num_samples):

        torch.cuda.empty_cache()
        gc.collect()
        
        prompt_text = construct_prompt(question, tokenizer)
        tokenized_prompt = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = tokenized_prompt["input_ids"].shape[-1]
        with torch.no_grad():
            output = model.generate(
                **tokenized_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

            generated_ids = output.sequences[0]
            answer_ids = generated_ids[prompt_len:]
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            final_answer = extract_answer(answer_text)
            answers.append(final_answer)
    
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return None, None, False, None
    
    counts = pd.Series(valid_answers).value_counts()
    majority_answer = counts.index[0]
    majority_freq = counts.iloc[0]

    resolved = False

    if not majority_answer or not ground_truth:
        resolved = False
    try:
        resolved = check_is_correct(majority_answer, ground_truth)
    except:
        resolved = False

    ppl_conf = compute_input_perplexity(model, tokenizer, question)

    return majority_answer, majority_freq, resolved, ppl_conf


def run(args):
    available_gpus = get_available_gpus(args.exclude_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in available_gpus])
    dataset = load_dataset(args.data_dir)

    questions = dataset["question"].tolist()
    questions = questions[: args.num_problems]
    answers = dataset["numeric_final_answer"].astype(str).tolist()
    answers = answers[: args.num_problems]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    problem_results = []

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    for q, gt in tqdm(zip(questions, answers), total=len(questions), desc="Evaluating problems"):
        majority_answer, freq, resolved, ppl_conf = run_single_problem(
            model, tokenizer, q, gt, args.num_samples, args.temperature, args.max_new_tokens
        )

        problem_results.append(
            {
                "question": q,
                "ground_truth": gt,
                "majority_answer": majority_answer,
                "majority_freq": freq,
                "resolved": resolved,
                "input_perplexity": ppl_conf,
            }
        )

    df = pd.DataFrame(problem_results)
    
    solved_conf = df[df["resolved"]]["input_perplexity"]
    unsolved_conf = df[~df["resolved"]]["input_perplexity"]

    print(f"Solved: {len(solved_conf)} | Unsolved: {len(unsolved_conf)}")
    print(f"Mean perplexity (Solved): {solved_conf.mean():.4f}")
    print(f"Mean perplexity (Unsolved): {unsolved_conf.mean():.4f}")

    if len(solved_conf) > 0 and len(unsolved_conf) > 0:
        ks_stat, p_value = ks_2samp(solved_conf, unsolved_conf)
        print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4e}")
    else:
        ks_stat, p_value = np.nan, np.nan

    plt.figure(figsize=(8, 4))
    bins = np.linspace(df["input_perplexity"].min(), df["input_perplexity"].max(), 50)
    plt.hist(solved_conf, bins=bins, alpha=0.7, color="royalblue", label="Solved")
    plt.hist(unsolved_conf, bins=bins, alpha=0.7, color="tomato", label="Unsolved")
    if len(solved_conf) > 0:
        plt.axvline(solved_conf.mean(), color="navy", ls="--", lw=1.5, label="Solved Mean")
    if len(unsolved_conf) > 0:
        plt.axvline(unsolved_conf.mean(), color="darkred", ls="--", lw=1.5, label="Unsolved Mean")
    plt.xlabel("Input Perplexity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Majority Voting | KS={ks_stat:.3f}, p={p_value:.2e}", fontsize=13)
    plt.legend()
    plt.tight_layout()

    save_dir = os.path.join("outputs/confidence_distribution", "input_perplexity")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "input_perplexity_plot.png"), dpi=300)
    plt.close()

    df.to_csv(os.path.join(save_dir, "input_perplexity_results.csv"), index=False)

    print("Finished majority voting and perplexity analysis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str,
                        default="data/src_custom_datasets_math_dataset_train_processed.parquet")
    parser.add_argument("--model", type=str,
                        default="/data/sunqiao/projects/models/meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=float, default=2048)
    parser.add_argument("--exclude_gpus", type=str, default="2, 3, 4, 5, 6, 7")
    args = parser.parse_args()
    run(args)
