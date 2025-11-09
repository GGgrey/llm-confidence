import gc
from tqdm import tqdm
from pathlib import Path

import torch

from src.config import Config, method_groups
from src.utils import load_model_and_tokenizer, load_and_sample_parquet_datasets, load_lingua_model, setup_tokenizer_padding_config, \
    batch_messages_creation, save_results_to_csv, print_final_accuracy_table, extract_answer, equal_func
from src.methods.cer import cer
from src.methods.self_consistency import self_consistency
from src.methods.p_true import p_true
from src.methods.predictive_entropy import predictive_entropy
from src.methods.likelihood import likelihood
from src.methods.perplexity import perplexity
from src.methods.topk_entropy import topk_entropy
from src.methods.window_entropy import window_entropy
from src.methods.key_confidence import key_confidence
from src.methods.hidden_svd import hidden_svd
from src.methods.attention_eigenvalue import attention_eigenvalue
from src.methods.compression_confidence import compression_confidence
from src.methods.attention_weighted_confidence import attention_weighted_confidence
from src.methods.trend_estimation import trend_estimation
from src.methods.attention_dynamic import attention_dynamic
from src.methods.group_entropy import group_entropy


def handle_sampling_group(
    model,
    tokenizer,
    lingua_model,
    batch_questions,
    tokenized_batch,
    config,
    group_cfgs,
    device
):
    
    batch_size = len(batch_questions)
    paths = [[] for _ in range(batch_size)]  # List for saving batch results
    group_results = []

    for _ in range(config.k):

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

        batch_output = model.generate(
            **tokenized_batch,
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

        for i in range(batch_size):
            generated_ids = batch_output.sequences[i]
            prompt_len = tokenized_batch["input_ids"][i].shape[0]
            answer_ids = generated_ids[prompt_len:]
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            output_scores = torch.stack([x[i] for x in batch_output.scores])

            final_answer = extract_answer(answer_text)

            paths[i].append({
                "prompt": batch_questions[i],
                "final_answer": final_answer,
                "generated_ids": generated_ids,
                "prompt_len": prompt_len,
                "answer_ids": answer_ids,
                "answer_text": answer_text,
                "output_scores": output_scores,
                "confidence": 1.0
            })

    # Retain valid path with final answer
    for batch_idx in range(batch_size):
        paths[batch_idx] = [p for p in paths[batch_idx] if p["final_answer"] is not None]

    # For debug
    if config.verbose:
        for i, sample_paths in enumerate(paths):
            for idx, path in enumerate(sample_paths):
                print(f"Batch {i}, sample {idx}: {path['answer_text']}")
    
    for method_name, method_cfg in group_cfgs.items():
        print(f"Current method: {method_name}, method config: {method_cfg}")
        method_result = []

        for i, sample_paths in enumerate(paths):
            if not sample_paths:
                method_result.append(("", 0.0, ""))
                continue

            if method_name == "cer_prob_product_log_last":
                method_output = cer(sample_paths, method_cfg, tokenizer, config)

            elif method_name == "cer_entropy_log_last":
                method_output = cer(sample_paths, method_cfg, tokenizer, config)

            elif method_name == "cer_entropy_weighted_mean_all":
                method_output = cer(sample_paths, method_cfg, tokenizer, config)

            elif method_name == "self_consistency":
                method_output = self_consistency(sample_paths)

            elif method_name == "p_true":
                method_output = p_true(sample_paths, batch_questions[i], tokenizer, model, device, config)

            elif method_name == "predictive_entropy":
                method_output = predictive_entropy(sample_paths, False, tokenizer, config)

            elif method_name == "normilized_entropy":
                method_output = predictive_entropy(sample_paths, True, tokenizer, config)

            elif method_name == "likelihood":
                method_output = likelihood(sample_paths, False, tokenizer, config)

            elif method_name == "normilized_likelihood":
                method_output = likelihood(sample_paths, True, tokenizer, config)

            elif method_name == "perplexity":
                method_output = perplexity(sample_paths, tokenizer, config)

            elif method_name == "topk_entropy":
                method_output = topk_entropy(sample_paths, tokenizer, config)

            elif method_name == "window_entropy":
                method_output = window_entropy(sample_paths, tokenizer, config)

            elif method_name == "key_confidence":
                method_output = key_confidence(sample_paths, method_cfg, model, config)
                
            elif method_name == "hidden_svd":
                method_output = hidden_svd(sample_paths, model, config)

            elif method_name == "attention_eigenvalue":
                method_output = attention_eigenvalue(sample_paths, method_cfg, model, tokenizer, config)

            elif method_name == "compression_confidence":
                method_output = compression_confidence(sample_paths, method_cfg, model, lingua_model, tokenizer, device, config)

            elif method_name == "attention_weighted_confidence":
                method_output = attention_weighted_confidence(sample_paths, tokenizer, model, config)

            elif method_name == "trend_estimation":
                method_output = trend_estimation(sample_paths, method_cfg, tokenizer, config)

            elif method_name == "attention_dynamic":
                method_output = attention_dynamic(sample_paths, method_cfg, model, tokenizer, config)

            elif method_name == "group_entropy":
                method_output = group_entropy(sample_paths, tokenizer, config)

            method_result.append(
                (method_output[0], method_output[1], method_output[2])
            )
        
        group_results.append({
            "method": method_name,
            "results": method_result
        })
    
    return group_results


def handle_greedy_group(
    model,
    tokenizer,
    batch_questions,
    tokenized_batch,
    config,
    group_cfgs,
    device
):
    
    batch_size = len(batch_questions)
    paths = [[] for _ in range(batch_size)]  # List for saving batch results
    group_results = []

    # Free memory
    torch.cuda.empty_cache()
    gc.collect()

    # Greedy sampling
    batch_output = model.generate(
        **tokenized_batch,
        max_new_tokens=config.max_new_tokens,
        num_beams=config.num_beams,
        temperature=config.temperature,
        do_sample=False,
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

    # Extract batch final answers
    for i in range(batch_size):
        generated_ids = batch_output.sequences[i]
        prompt_len = tokenized_batch["input_ids"][i].shape[0]
        answer_ids = generated_ids[prompt_len:]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        output_scores = torch.stack([x[i] for x in batch_output.scores])

        final_answer = extract_answer(answer_text)

        paths[i].append({
            "prompt": batch_questions[i],
            "final_answer": final_answer,
            "generated_ids": generated_ids,
            "prompt_len": prompt_len,
            "answer_ids": answer_ids,
            "answer_text": answer_text,
            "output_scores": output_scores,
            "confidence": 1.0
        })

    for method_name, method_cfg in group_cfgs.items():
        print(f"Current method: {method_name}, method config: {method_cfg}")
        method_result = []

        for i, sample_paths in enumerate(paths):
            if not sample_paths:
                method_result.append(("", 0.0, ""))
                continue

            if method_name == "greedy":
                last_sample = sample_paths[-1]
                method_result.append(
                    (last_sample["answer_text"], last_sample["confidence"], last_sample["final_answer"])
                )
            else:
                raise ValueError(f"Unsupported method: {method_name}")
        
        group_results.append({
            "method": method_name,
            "results": method_result
        })

    return group_results


def evaluate_batch_examples(
    model,
    tokenizer,
    lingua_model,
    batch_questions,
    batch_correct_answers,
    config
):
    all_results = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure tokenizer's padding token and padding side
    setup_tokenizer_padding_config(tokenizer, model)

    tokenized_batch = batch_messages_creation(
        tokenizer,
        batch_questions,
        device,
        config.use_base_prompt
    )

    # Run each configuration, with different sampling parameters for different configurations
    for group_name, group_cfgs in method_groups.items():
        print(f"\nGroup name: {group_name}")

        if group_name == "greedy":
            group_results = handle_greedy_group(model, tokenizer, batch_questions, tokenized_batch, config, group_cfgs, device)
        elif group_name == "sampling":
            group_results = handle_sampling_group(model, tokenizer, lingua_model, batch_questions, tokenized_batch, config, group_cfgs, device)
        
        all_results.append(group_results)

    all_method_results = []

    for group_idx, group_results in enumerate(all_results, start=1):
        print(f"\n=== Group {group_idx} ===")
        for method_idx, method_result in enumerate(group_results, start=1):
            method_name = method_result["method"]
            print(f"Method {method_idx}: {method_name}")

            method_evaluation_result = []

            for batch_idx, (text, score, answer) in enumerate(method_result["results"], start=1):
                print(f"  Batch {batch_idx}:")
                print(f"    Text: {text[:80]}...")
                print(f"    Score: {score}")
                print(f"    Answer: {answer}")

                ground_truth = batch_correct_answers[batch_idx - 1]
                print(f"    Ground truth: {ground_truth}")

                is_correct = False

                if not answer or not ground_truth:
                    is_correct = False

                try:
                    is_correct = equal_func(answer, ground_truth)
                except:
                    is_correct = str(answer) == str(ground_truth)
            
                method_evaluation_result.append({
                    "method_name": method_name,
                    "question": batch_questions[batch_idx - 1],
                    "ground_truth": ground_truth,
                    "final_answer": answer,
                    "answer_text": text,
                    "confidence": score,
                    "is_correct": is_correct
                })
        
            all_method_results.append(method_evaluation_result)
    
    return all_method_results


def evaluate_dataset(
    model,
    tokenizer,
    lingua_model,
    dataset,
    config,
    dataset_name
):
    # Extract lists of questions and answers directly from the dataframe
    questions = dataset["question"].tolist()

    correct_answers_list = dataset["numeric_final_answer"].astype(str).tolist()
    
    total_questions = len(questions)
    description=f"{dataset_name}_{config.model_name.replace('/', '_')}"
    correct_answers = {}
    method_names = []
    all_results = {}

    # Process the dataset in batches
    with tqdm(total=total_questions, desc=f"Processing {description}", dynamic_ncols=True) as pbar:
        for start_idx in range(0, total_questions, config.batch_size):
            end_idx = min(start_idx + config.batch_size, total_questions)

            # Slice out the batch
            batch_questions = questions[start_idx:end_idx]
            batch_correct_answers = correct_answers_list[start_idx:end_idx]

            # Evaluate the batch
            batch_results = evaluate_batch_examples(
                model,
                tokenizer,
                lingua_model,
                batch_questions,
                batch_correct_answers,
                config
            )

            if config.verbose:
                print(f"\nBatch results: {batch_results}")

            for method_batch_results in batch_results:

                method_name = method_batch_results[0]["method_name"]

                if method_name not in correct_answers:
                    correct_answers[method_name] = 0
                    all_results[method_name] = []
                    method_names.append(method_name)

                batch_correct_count = sum(1 for r in method_batch_results if r["is_correct"])
                correct_answers[method_name] += batch_correct_count
                all_results[method_name].extend(method_batch_results)

            postfix_dict = {}
            print("\n" + "=" * 50)
            for m in method_names:
                method_running_accuracy = (correct_answers[m] / end_idx) * 100
                print(f"Method name: {m}, running accuracy: {method_running_accuracy}")
                postfix_dict[f"{m}_acc"] = f"{method_running_accuracy:.2f}%"
            print("=" * 50)
            
            pbar.set_postfix(postfix_dict)
            pbar.update(end_idx - start_idx)

    directory_path = Path("outputs")
    directory_path.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(
        all_results,
        f"{directory_path}/{description}_evaluation_results_zero_shot.csv"
    )

    method_final_accuracy = {
        m: (correct_answers[m] / total_questions) * 100
        for m in method_names
    }

    print_final_accuracy_table(method_final_accuracy)
                    

def run(config):
    """
    Run the complete dataset evaluation pipeline based on the provided configuration.

    Args:
        config (Config): A configuration object.
    """
    # Print the provided configurations
    print("=" * 50)
    print("Configurations:")
    print(f"Model name: {config.model_name}")
    print(f"Lingua model name: {config.lingua_model_name}")
    print(f"Aggregate: {config.aggregate}")
    print(f"k: {config.k}")
    print(f"Number of samples: {config.number_samples}")
    print(f"Seed: {config.seed}")
    print(f"Data directory: {config.data_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Dataset files: {config.datasets}")
    print("=" * 50 + "\n")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_name, config.read_model_from_huggingface)

    # Load dataset for evaluation
    datasets = load_and_sample_parquet_datasets(config.data_dir, config.datasets,
                                                number_samples=config.number_samples,
                                                seed=config.seed)
    
    # Load lingua model for text compression
    lingua_model = load_lingua_model(config.lingua_model_name)

    # Evaluate methods on each dataset
    for dataset_name, dataset_df in datasets.items():
        print(f"\nDataset name: {dataset_name}")

        evaluate_dataset(
            model,
            tokenizer,
            lingua_model,
            dataset_df,
            config,
            dataset_name,
        )

        print(f"\nFinished evaluating: {dataset_name}")
