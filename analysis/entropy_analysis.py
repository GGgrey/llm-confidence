import gc
import os

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
import numpy as np

from src.config import Config
from src.utils.utils import load_model_and_tokenizer, setup_tokenizer_padding_config, construct_prompt, extract_answer
from src.utils.grader import math_equal, check_is_correct
from src.utils.parser import extract_answer_pro


def load_dataset(file_path):
    df = None
    if os.path.isfile(file_path):
        df = pd.read_parquet(file_path)
    return df


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


def trend_theilsen(entropy):
    x = np.arange(len(entropy)).reshape(-1, 1)
    y = np.array(entropy)
    model = TheilSenRegressor(random_state=0)
    model.fit(x, y)
    slope = model.coef_[0]
    trend = "upward" if slope > 0 else "downward"
    return slope, trend


def trend_savgol(entropy, window=7, poly=3):
    smooth_entropy = savgol_filter(entropy, window_length=min(window, len(entropy)), polyorder=poly)
    diff = np.diff(smooth_entropy)
    mean_diff = np.mean(diff)
    trend = "upward" if mean_diff > 0 else "downward"
    return mean_diff, trend, smooth_entropy


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


config = Config()

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    config.model_name,
    config.read_model_from_huggingface
)

# Load dataset
dataset = load_dataset(
    "data/src_custom_datasets_math_dataset_test_processed.parquet"
)

# Extract lists of questions and answers directly from the dataframe
questions = dataset["question"].tolist()
correct_answers_list = dataset["numeric_final_answer"].astype(str).tolist()
total_questions = len(questions)

device = "cuda" if torch.cuda.is_available() else "cpu"

with tqdm(total=total_questions, desc=f"Processing math dataset", dynamic_ncols=True) as pbar:
    for q_idx in range(total_questions):
        question = questions[q_idx]
        ground_truth = correct_answers_list[q_idx]

        # Configure tokenizer's padding token and padding side
        setup_tokenizer_padding_config(tokenizer, model)

        message = {
            "role": "user",
            "content": construct_prompt(
                question=question,
                use_base_prompt=config.use_base_prompt
            )
        }

        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            input_text = tokenizer.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in message
            ])
            input_text += "\nassistant:"

        tokenized_input = tokenizer(
            [input_text],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        for sample_idx in range(config.k):
            
            # Free memory
            torch.cuda.empty_cache()
            gc.collect()

            batch_output = model.generate(
                **tokenized_input,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                temperature=config.temperature,
                do_sample=True,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=config.early_stopping,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

            generated_ids = batch_output.sequences[0]
            prompt_len = tokenized_input["input_ids"][0].shape[0]
            answer_ids = generated_ids[prompt_len:]
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            output_scores = torch.stack([x[0] for x in batch_output.scores])

            final_answer = extract_answer(answer_text)
            if final_answer is None:
                continue

            try:
                is_correct = check_is_correct(final_answer, ground_truth)
            except:
                is_correct = str(final_answer) == str(ground_truth)

            print("=" * 100)
            print(f"\nQuestion id: {q_idx}, sample id: {sample_idx}")
            print(f"\nQuestion: {question}")
            print(f"\nGround truth: {ground_truth}")
            print(f"\nAnswer text: {answer_text}")
            print(f"\nFinal answer: {final_answer}")
            print(f"\nIs correct: {is_correct}")

            # Calculate token probs
            probs = F.softmax(output_scores, dim=-1)

            # Likelihood
            generated_probs = []
            for token_idx, token_id in enumerate(answer_ids):
                token_prob = probs[token_idx, token_id].item()
                generated_probs.append(token_prob)
            generated_probs = torch.tensor(generated_probs).cpu().numpy()

            # Entropy
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).cpu().numpy()

            # Calculate entropy trend
            slope_lr, intercept, r2, trend_lr = trend_linear_regression(entropy)
            mean_diff, trend_sg, smooth_entropy = trend_savgol(entropy)
            slope_ts, trend_ts = trend_theilsen(entropy)

            # Get attention weights
            attn_weights = get_attn_weights(generated_ids, model)
            num_layers = len(attn_weights)
            num_heads = attn_weights[0].shape[1]

            # Attention entropy
            attn_entropy_matrix = np.zeros((num_layers, num_heads, len(generated_ids) - prompt_len))
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    full_attn = attn_weights[layer_idx][0, head_idx].to(torch.float).cpu().numpy()  # (seq, seq)
                    for idx, token_idx in enumerate(range(prompt_len, len(generated_ids))):
                        p = full_attn[token_idx, :token_idx+1] + 1e-10
                        p /= p.sum()
                        attn_entropy_matrix[layer_idx, head_idx, idx] = -(p * np.log(p)).sum()
            
            attn_entropy_per_token = attn_entropy_matrix.mean(axis=(0, 1))
            attn_entropy = attn_entropy_per_token.mean()

            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(20, 8), sharex=True,
                gridspec_kw={'height_ratios': [1, 1, 1]}
            )

            color1 = '#1f77b4'
            ax1.plot(range(1, len(entropy) + 1), entropy, color='#1f77b4', linewidth=2, label="Entropy")
            ax1.set_ylabel("Entropy", color=color1, fontsize=12, fontweight="bold")
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, linestyle="--", alpha=0.4)
            ax1.legend(loc='upper right')

            color2 = '#ff7f0e'
            ax2.plot(range(1, len(generated_probs) + 1), generated_probs, color=color2, linewidth=2, label="Token Probability")
            ax2.set_ylabel('Token Probability', color=color2, fontsize=12, fontweight="bold")
            ax2.set_xlabel("Token Position", fontsize=12, fontweight="bold")
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.grid(True, linestyle="--", alpha=0.4)
            ax2.legend(loc='upper right')

            color3 = '#2ca02c'
            ax3.plot(range(1, len(attn_entropy_per_token) + 1), attn_entropy_per_token, color=color3, linewidth=2, label="Attention Entropy")
            ax3.set_ylabel("Attention Entropy", color=color3, fontsize=12, fontweight="bold")
            ax3.set_xlabel("Token Position", fontsize=12, fontweight="bold")
            ax3.tick_params(axis='y', labelcolor=color3)
            ax3.grid(True, linestyle="--", alpha=0.4)
            ax3.legend(loc='upper right')

            fig.text(0.5, 0.99, 
                f"Entropy & Generated Token Prob (Q{q_idx} Sample: {sample_idx}, lr: {slope_lr}, attn_entropy: {attn_entropy})\nCorrect: {is_correct}",
                ha='center', va='top',
                fontsize=14, fontweight="bold"
            )

            base_dir = os.path.join("outputs/plots_1/", f"Q{q_idx}")
            save_dir = os.path.join(base_dir, "correct" if is_correct else "wrong")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"q{q_idx}_sample{sample_idx}_{'correct' if is_correct else 'wrong'}.png"
            )
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print("\n" + "=" * 100)
        
        pbar.update(1)

