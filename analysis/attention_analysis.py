import gc
import os

from matplotlib import gridspec
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
import numpy as np
import seaborn as sns

from src.config import Config
from src.utils.utils import load_model_and_tokenizer, setup_tokenizer_padding_config, construct_prompt, extract_answer
from src.utils.grader import math_equal, check_is_correct


def load_dataset(file_path):
    df = None
    if os.path.isfile(file_path):
        df = pd.read_parquet(file_path)
    return df


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
    "data/openai_gsm8k_test_processed.parquet"
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
                add_generation_prompt=True,
                enable_thinking=False
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
                max_new_tokens=1024,
                num_beams=config.num_beams,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
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

            # Get attention weights
            attn_weights = get_attn_weights(generated_ids, model)
            num_layers = len(attn_weights)
            num_heads = attn_weights[0].shape[1]

            seq_len = len(generated_ids)
            generated_token_indices = range(prompt_len, seq_len)

            d_matrix = np.zeros((num_layers, num_heads))
            all_attn_maps = {}
            
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    full_attn = attn_weights[layer_idx][0, head_idx].to(torch.float).cpu().numpy()
                    all_attn_maps[(layer_idx, head_idx)] = full_attn

                    backward_distances = []
                    for t in generated_token_indices:
                        p = full_attn[t, :t+1]
                        p = p / p.sum()

                        distances = np.arange(t, -1, -1)

                        weighted_dist = np.sum(p * distances)
                        backward_distances.append(weighted_dist)
                    
                    d_matrix[layer_idx, head_idx] = np.mean(backward_distances)

            # Local and global attention map
            flattened_d = d_matrix.flatten()
            num_heads_total = len(flattened_d)

            sorted_indices = np.argsort(flattened_d)
            bottom_k = int(0.3 * num_heads_total)
            top_k = int(0.3 * num_heads_total)

            local_heads_idx = sorted_indices[:bottom_k]
            global_heads_idx = sorted_indices[-top_k:]

            H_loc = [(idx // num_heads, idx % num_heads) for idx in local_heads_idx]
            H_glob = [(idx // num_heads, idx % num_heads) for idx in global_heads_idx]

            A_loc_sum = np.zeros_like(next(iter(all_attn_maps.values())))

            for (l, h) in H_loc:
                A_loc_sum += all_attn_maps[(l, h)]
                A_loc_avg = A_loc_sum / len(H_loc)

            A_glob_sum = np.zeros_like(next(iter(all_attn_maps.values())))
            for (l, h) in H_glob:
                A_glob_sum += all_attn_maps[(l, h)]
                A_glob_avg = A_glob_sum / len(H_glob)

            # WAAD
            window_size = 10
            gen_matrix_loc = A_loc_avg[prompt_len:, prompt_len:]
            seq_gen = gen_matrix_loc.shape[0]
            waad_values = np.zeros(seq_gen)

            for t in range(seq_gen):
                for s in range(t+1):
                    dist = min(t - s, window_size)
                    waad_values[t] += gen_matrix_loc[t, s] * dist

            # FAI
            H_lo = 10
            H_hi = 50
            gen_matrix_glob = A_glob_avg[prompt_len:, prompt_len:]
            fai_values = np.zeros(seq_gen)

            for s in range(seq_gen):
                start_t = s + H_lo
                end_t = min(seq_gen, s + H_hi)
                if start_t < end_t:
                    t_range = range(start_t, end_t)
                    fai_values[s] = np.mean([gen_matrix_glob[t, s] for t in t_range])
                else:
                    fai_values[s] = 0.0

            # Plotting
            base_dir = os.path.join("outputs/attention_analysis/", f"Q{q_idx}")
            save_dir = os.path.join(base_dir, "correct" if is_correct else "wrong")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"q{q_idx}_sample{sample_idx}_{'correct' if is_correct else 'wrong'}.png"
            )

            fig = plt.figure(figsize=(14,10))
            gs = gridspec.GridSpec(4, 2, height_ratios=[3, 1, 1, 1])

            ax1 = fig.add_subplot(gs[0, 0])
            sns.heatmap(A_loc_avg, ax=ax1, vmin=0.0, vmax=0.14, cmap="viridis")
            ax1.set_title("Aggregated Attention (Local-focused)")

            ax2 = fig.add_subplot(gs[0, 1])
            sns.heatmap(A_glob_avg, ax=ax2, vmin=0, vmax=0.1, cmap="viridis")
            ax2.set_title("Aggregated Attention (Global-focused)")

            ax3 = fig.add_subplot(gs[1, :])
            ax3.plot(waad_values, label="WAAD", color='#1f77b4', alpha=0.8)
            ax3.set_title(f"WAAD curve (W={window_size})")
            ax3.set_xlabel("Generated token index")
            ax3.set_ylabel("WAAD value")
            ax3.grid(True)
            ax3.legend()

            ax4 = fig.add_subplot(gs[2, :])
            ax4.plot(fai_values, label=f"FAI", color='#ff7f0e', alpha=0.8)
            ax4.set_title(f"FAI curve (H_lo={H_lo}, H_hi={H_hi})")
            ax4.set_xlabel("Generated token index")
            ax4.set_ylabel("FAI value")
            ax4.grid(True)
            ax4.legend()

            ax5 = fig.add_subplot(gs[3, :])
            ax5.plot(entropy, label=f"Token Entropy", color='#2ca02c', alpha=0.8)
            ax5.set_title(f"Token Entropy")
            ax5.set_xlabel("Generated token index")
            ax5.set_ylabel("Token Entropy")
            ax5.grid(True)
            ax5.legend()

            plt.tight_layout()
            plt.show()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print("\n" + "=" * 100)
        
        pbar.update(1)

