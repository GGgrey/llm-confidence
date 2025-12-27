import gc
from tqdm import tqdm
from pathlib import Path
import os

import torch

from src.config import method_groups
from src.utils.utils import (
    load_model_and_tokenizer, load_and_sample_parquet_datasets, load_lingua_model, save_jsonl, save_results_to_json,
    setup_tokenizer_padding_config, batch_messages_creation, save_results_to_csv,
    print_final_accuracy_table, extract_answer, load_datasets,
    seed_everything, get_available_gpus
)
from src.utils.grader import math_equal, check_is_correct
from src.utils.parser import extract_answer_pro, extract_answer
from src.methods.oracle import oracle_self_consistency
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
from src.methods.quantile import quantile
from src.methods.xentropy import xentropy
from src.methods.stability_aware_entropy import stability_aware_entropy
from src.methods.heterogeneous_ensemble import heterogeneous_ensemble
from src.methods.semantic_consistency import semantic_consistency
from src.methods.distinct_entropy import distinct_entropy
from src.methods.permutation_entropy import permutation_entropy
from src.methods.self_certainty import self_certainty
from src.methods.energy import energy
from src.methods.logit import logit_scoring
from src.methods.logtoku import logtoku
from src.methods.deepconf import deepconf
from src.methods.stable_rank import stable_rank
from src.methods.attention_entropy import attention_entropy
from src.methods.generalized_entropy import generalized_entropy
from src.methods.trajectory_pattern import trajectory_pattern
from src.methods.anisotropy_evolution import anisotropy_evolution
from src.methods.hidden_complexity import hidden_complexity
from src.methods.rank_weighted_confidence import rank_weighted_confidence


def dispatch_method(
    method_name,
    method_cfg,
    sample_paths,
    tokenizer,
    model,
    lingua_model,
    device,
    config,
):
    if method_name == "oracle":
        return oracle_self_consistency(sample_paths)

    elif method_name.startswith("cer_"):
        return cer(sample_paths, method_cfg, tokenizer, config)

    elif method_name == "self_consistency":
        return self_consistency(sample_paths)

    elif method_name == "p_true":
        return p_true(sample_paths, tokenizer, model, device, config)

    elif method_name == "predictive_entropy":
        return predictive_entropy(sample_paths, False, tokenizer, config)

    elif method_name == "normalized_entropy":
        return predictive_entropy(sample_paths, True, tokenizer, config)

    elif method_name == "likelihood":
        return likelihood(sample_paths, False, tokenizer, config)

    elif method_name == "normalized_likelihood":
        return likelihood(sample_paths, True, tokenizer, config)

    elif method_name == "perplexity":
        return perplexity(sample_paths, tokenizer, config)

    elif method_name == "topk_entropy":
        return topk_entropy(sample_paths, tokenizer, config)

    elif method_name == "window_entropy":
        return window_entropy(sample_paths, tokenizer, config)

    elif method_name == "key_confidence":
        return key_confidence(sample_paths, method_cfg, model, tokenizer, config)
                
    elif method_name == "hidden_svd":
        return hidden_svd(sample_paths, model, tokenizer, config)

    elif method_name == "attention_eigenvalue":
        return attention_eigenvalue(sample_paths, method_cfg, model, tokenizer, config)

    elif method_name == "compression_confidence":
        return compression_confidence(sample_paths, method_cfg, model, lingua_model, tokenizer, device, config)

    elif method_name == "attention_weighted_confidence":
        return attention_weighted_confidence(sample_paths, tokenizer, model, config)

    elif method_name == "trend_estimation":
        return trend_estimation(sample_paths, method_cfg, tokenizer, config)

    elif method_name == "attention_dynamic":
        return attention_dynamic(sample_paths, method_cfg, model, tokenizer, config)

    elif method_name == "group_entropy":
        return group_entropy(sample_paths, method_cfg, tokenizer, config)

    elif method_name.startswith("quantile_"):
        return quantile(sample_paths, method_cfg, tokenizer, config)

    elif method_name == "gibbs_entropy_lin" or method_name == "gibbs_entropy_exp" or \
        method_name == "tsallis_entropy_lin" or method_name == "tsallis_entropy_exp" or \
        method_name == "renyi_entropy_lin" or method_name == "renyi_entropy_exp":
        return xentropy(sample_paths, method_cfg, tokenizer, config)
    
    elif method_name == "stability_aware_entropy":
        return stability_aware_entropy(sample_paths, method_cfg, tokenizer, config)
    
    elif method_name.startswith("heterogeneous_ensemble_"):
        return heterogeneous_ensemble(sample_paths, method_cfg, tokenizer, config)
    
    elif method_name == "semantic_consistency":
        return semantic_consistency(sample_paths, config)
    
    elif method_name == "distinct_entropy":
        return distinct_entropy(sample_paths, method_cfg, True, tokenizer, config)
    
    elif method_name == "permutation_entropy":
        return permutation_entropy(sample_paths, tokenizer, config)
    
    elif method_name == "self_certainty":
        return self_certainty(sample_paths, method_cfg, tokenizer, config)
    
    elif method_name.startswith("energy_"):
        return energy(sample_paths, method_cfg, tokenizer, config)
    
    elif method_name.startswith("logit_"):
        return logit_scoring(sample_paths, method_cfg, tokenizer, config)
    
    elif method_name == "logtoku":
        return logtoku(sample_paths, method_cfg, tokenizer, config)
    
    elif method_name.startswith("deepconf_"):
        return deepconf(sample_paths, method_cfg, tokenizer, config)
    
    elif method_name == "stable_rank":
        return stable_rank(sample_paths, model, tokenizer, config)
    
    elif method_name == "attention_entropy":
        return attention_entropy(sample_paths, method_cfg, model, tokenizer, config)
    
    elif method_name == "generalized_entropy":
        return generalized_entropy(sample_paths, method_cfg, tokenizer, config)
    
    elif method_name == "trajectory_pattern":
        return trajectory_pattern(sample_paths, model, tokenizer, config)
    
    elif method_name == "anisotropy_evolution":
        return anisotropy_evolution(sample_paths, model, tokenizer, config)
    
    elif method_name == "hidden_complexity":
        return hidden_complexity(sample_paths, model, tokenizer, config)
    
    elif method_name == "rank_weighted_confidence":
        return rank_weighted_confidence(sample_paths, model, tokenizer, config)

    else:
        raise ValueError(f"Unsupported method: {method_name}")


def handle_sampling_group(
    model,
    tokenizer,
    lingua_model,
    batch_questions,
    batch_correct_answers,
    tokenized_batch,
    config,
    group_cfgs,
    device
):
    
    batch_size = len(batch_questions)
    paths = [[] for _ in range(batch_size)]  # List for saving batch results
    group_results = []
    group_records = []

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
            use_cache=True,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
        )

        for i in range(batch_size):
            generated_ids = batch_output.sequences[i].detach().cpu()
            prompt_len = tokenized_batch["input_ids"][i].shape[0]
            answer_ids = generated_ids[prompt_len:].detach().cpu()
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            output_scores = torch.stack([x[i] for x in batch_output.scores]).detach().cpu()
            output_logits = torch.stack([x[i] for x in batch_output.logits]).detach().cpu()

            final_answer = extract_answer(answer_text)

            paths[i].append({
                "prompt": batch_questions[i],
                "ground_truth": batch_correct_answers[i],
                "final_answer": final_answer,
                "generated_ids": generated_ids,
                "prompt_len": prompt_len,
                "answer_ids": answer_ids,
                "answer_text": answer_text,
                "output_scores": output_scores,
                "output_logits": output_logits,
                "confidence": 1.0
            })
        
        del batch_output

    # Retain valid path with final answer
    for batch_idx in range(batch_size):
        paths[batch_idx] = [p for p in paths[batch_idx] if p["final_answer"]]

    # For debug
    if config.verbose:
        for i, sample_paths in enumerate(paths):
            for idx, path in enumerate(sample_paths):
                print(f"Batch {i}, sample {idx}: {path['answer_text']}")
    
    for method_name, method_cfg in group_cfgs.items():
        print(f"Current method: {method_name}, method config: {method_cfg}")
        method_result = []
        record = []

        for i, sample_paths in enumerate(paths):
            if not sample_paths:
                method_result.append(("", 0.0, ""))
                continue

            method_output = dispatch_method(
                method_name, method_cfg,
                sample_paths,
                tokenizer,
                model, lingua_model,
                device,
                config
            )

            method_result.append(
                (method_output[0], method_output[1], method_output[2])
            )

            record.append(method_output[3])
        
        group_results.append({
            "method": method_name,
            "results": method_result
        })

        group_records.append({
            "method": method_name,
            "records": record
        })

    batch_info = []
    for i in range(batch_size):
        batch_info.append({
            "question": batch_questions[i],
            "ground_truth": batch_correct_answers[i],
            "k": config.k,
            "samples": [
                {
                    "sample_idx": j,
                    "answer_text": p.get("answer_text", ""),
                    "final_answer": p.get("final_answer", None),
                }
                for j, p in enumerate(paths[i])
            ],
            "records": [
                {
                    "method": gr["method"],
                    "record": gr["records"][i] if i < len(gr["records"]) else None
                }
                for gr in group_records
            ]
        })
    
    return group_results, batch_info


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
        use_cache=True,
        output_scores=True,
        # output_logits=True,
        return_dict_in_generate=True,
    )

    # Extract batch final answers
    for i in range(batch_size):
        generated_ids = batch_output.sequences[i].detach().cpu()
        prompt_len = tokenized_batch["input_ids"][i].shape[0]
        answer_ids = generated_ids[prompt_len:].detach().cpu()
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        output_scores = torch.stack([x[i] for x in batch_output.scores]).detach().cpu()
        # output_logits = torch.stack([x[i] for x in batch_output.logits])

        final_answer = extract_answer(answer_text)

        paths[i].append({
            "prompt": batch_questions[i],
            "final_answer": final_answer,
            "generated_ids": generated_ids,
            "prompt_len": prompt_len,
            "answer_ids": answer_ids,
            "answer_text": answer_text,
            "output_scores": output_scores,
            # "output_logits": output_logits,
            "confidence": 1.0
        })

    del batch_output

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
    all_batch_info = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(config.seed)

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
            group_results = handle_greedy_group(
                model, tokenizer,
                batch_questions,
                tokenized_batch,
                config, group_cfgs, device
            )
        elif group_name == "sampling":
            group_results, batch_info = handle_sampling_group(
                model, tokenizer, lingua_model,
                batch_questions, batch_correct_answers,
                tokenized_batch,
                config, group_cfgs, device
            )
            all_batch_info.extend(batch_info)
        
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
                print(f"    Question: {batch_questions[batch_idx - 1]}")
                print(f"    Answer text: {text[:160]}...")
                print(f"    Answer length: {len(text)}")
                print(f"    Score: {score}")
                print(f"    Final answer: {answer}")

                ground_truth = batch_correct_answers[batch_idx - 1]
                print(f"    Ground truth: {ground_truth}")

                is_correct = False

                if not answer or not ground_truth:
                    is_correct = False

                try:
                    is_correct = check_is_correct(answer, ground_truth)
                except:
                    is_correct = False
                print(f"    Result: {'Correct' if is_correct else 'Wrong'}")
            
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
    
    return all_method_results, all_batch_info


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
    answers = dataset["answer"].astype(str).tolist()
    
    total_questions = len(questions)
    # total_questions = 1

    model_tag = str(config.model_name).rstrip("/").split("/")[-1]
    description=f"{dataset_name}_{model_tag}"
    correct_answers, method_names, all_results = {}, [], {}
    all_batch_info = []

    # Process the dataset in batches
    with tqdm(total=total_questions, desc=f"Processing {description}", dynamic_ncols=True) as pbar:
        for start_idx in range(0, total_questions, config.batch_size):
            end_idx = min(start_idx + config.batch_size, total_questions)

            # Slice out the batch
            batch_questions = questions[start_idx:end_idx]
            batch_correct_answers = answers[start_idx:end_idx]

            # Evaluate the batch
            batch_results, batch_info = evaluate_batch_examples(
                model,
                tokenizer,
                lingua_model,
                batch_questions,
                batch_correct_answers,
                config
            )

            all_batch_info.extend(batch_info)

            if config.verbose:  # For debug
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

    method_final_accuracy = {
        m: (correct_answers[m] / total_questions) * 100
        for m in method_names
    }

    print_final_accuracy_table(method_final_accuracy)

    payload = {
        "meta": {
            "dataset_name": dataset_name,
            "description": description,
            "model_name": config.model_name,
            "lingua_model_name": getattr(config, "lingua_model_name", None),
            "seed": config.seed,
            "k": config.k,
            "batch_size": config.batch_size,
            "max_new_tokens": config.max_new_tokens,
            "num_beams": config.num_beams,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "length_penalty": config.length_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "early_stopping": config.early_stopping,
            "use_base_prompt": getattr(config, "use_base_prompt", None),
            "aggregate": config.aggregate,
            "total_questions": total_questions,
        },
        "summary": {
            "method_final_accuracy": method_final_accuracy,
            "correct_counts": correct_answers,
        },
        "results": all_results,
    }

    directory_path = Path("outputs")
    directory_path.mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(
        directory_path,
        f"{description}_k{config.k}_seed{config.seed}_t{config.temperature}_tp{config.top_p}_evaluation_results.jsonl"
    )
    save_results_to_json(payload, save_path)

    samples_payload = {
        "meta": payload["meta"],
        "samples": all_batch_info,
    }
    samples_save_path = os.path.join(
        directory_path,
        f"{description}_k{config.k}_seed{config.seed}_t{config.temperature}_tp{config.top_p}_all_samples.json"
    )
    save_results_to_json(samples_payload, samples_save_path)


def run(config):

    print("=" * 50)
    print("Configurations:")
    print(f"Reasoning model name: {config.model_name}")
    print(f"Lingua model name: {config.lingua_model_name}")
    print(f"Aggregate: {config.aggregate}")
    print(f"K: {config.k}")
    # print(f"Number of samples: {config.number_samples}")
    print(f"Seed: {config.seed}")
    print(f"Data directory: {config.data_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Dataset files: {config.datasets}")
    print(f"Exclude gpus: {config.exclude_gpus}")
    print("=" * 50 + "\n")

    # Load dataset for evaluation
    # datasets = load_and_sample_parquet_datasets(
    #     config.data_dir,
    #     config.datasets,
    #     number_samples=config.number_samples,
    #     seed=config.seed
    # )
    datasets = load_datasets(config.data_dir, config.datasets)

    # Set available GPUs
    available_gpus = get_available_gpus(config.exclude_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in available_gpus])

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_name, config.read_model_from_huggingface)
    
    # Load lingua model for text compression
    lingua_model = None
    for group_name, group_cfgs in method_groups.items():
        if "compression_confidence" in group_cfgs:
            print(f"Compression confidence method exists in {group_name}")
            lingua_model = load_lingua_model(config.lingua_model_name)
            break
        
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
