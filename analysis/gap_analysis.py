import ast
import gc
import os
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.utils import construct_prompt, extract_answer, load_datasets, load_model_and_tokenizer, seed_everything, setup_tokenizer_padding_config
from src.utils.grader import check_is_correct, evaluate_math
from src.utils.parser import extract_answer_pro


def clean_and_parse(cell_value):
    if pd.isna(cell_value):
        return None
    
    cell_value = re.sub(r'np\.float64\((.*?)\)', r'\1', str(cell_value))
    
    try:
        return ast.literal_eval(cell_value)
    except Exception as e:
        print(f"Parse error: {e} for content snippet: {cell_value[:50]}")
        return None


file_path = './outputs/math_500__models_meta-llama_Llama-3.1-8B-Instruct_evaluation_results_zero_shot.csv'

df = pd.read_csv(file_path)

correctness_data = {}
methods = [col for col in df.columns if col != 'index']

for col in methods:
    parsed_col = df[col].apply(clean_and_parse)
    correctness_data[col] = parsed_col.apply(lambda x: x.get('is_correct', False) if x else False)

correctness_df = pd.DataFrame(correctness_data)
oracle_results = correctness_df['oracle']
other_methods = [col for col in correctness_df.columns if col != 'oracle']

method_failures = {}
for method in correctness_df.columns:
    failed_indices = correctness_df.index[~correctness_df[method]].tolist()
    method_failures[method] = failed_indices

others_all_wrong = ~correctness_df[other_methods].any(axis=1)
gap_indices = correctness_df.index[oracle_results & others_all_wrong].tolist()

print(f"Total questions: {len(df)}")
print(f"Oracle is correct, but all other methods are wrong: {len(gap_indices)}")
print(f"Gap question index: {gap_indices}")

for method, failures in method_failures.items():
    accuracy = 1 - (len(failures) / len(df))
    print(f"Method: {method:<30} | Wrong count: {len(failures):<4} | Accuracy: {accuracy:.2%}")

dataset = pd.read_json("data/math_500.jsonl", lines=True)
questions = dataset["question"].tolist()
answers = dataset["answer"].astype(str).tolist()
total_questions = len(questions)
question = questions[8]
answer = answers[8]
print(f"Question: {question}, ground truth: {answer}")

model, tokenizer = load_model_and_tokenizer(
    "/models/meta-llama/Llama-3.1-8B-Instruct", True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(10)

setup_tokenizer_padding_config(tokenizer, model)

content = construct_prompt(
    question=question,
    use_base_prompt=True
)
message = [{"role": "user", "content": content}]
input_text = tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True
)

tokenized_input = tokenizer(
    input_text,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
tokenized_input = tokenized_input.to(device)

paths = []
for sample_idx in range(16):

    torch.cuda.empty_cache()
    gc.collect()

    output = model.generate(
        **tokenized_input,
        max_new_tokens=2048,
        num_beams=1,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.0,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        early_stopping=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    generated_ids = output.sequences[0]
    prompt_len = tokenized_input["input_ids"][0].shape[0]
    answer_ids = generated_ids[prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    output_scores = torch.stack([x[0] for x in output.scores])

    final_answer = extract_answer(answer_text)

    if not final_answer:
        continue

    is_correct = check_is_correct(final_answer, answer)
    if not is_correct and "√" in final_answer:
        fixed_answer = re.sub(r'√\s*([0-9]+|[a-zA-Z])', r'\\sqrt{\1}', final_answer)
        fixed_answer = re.sub(r'√\s*\((.*?)\)', r'\\sqrt{\1}', fixed_answer)
        fixed_answer = fixed_answer.replace("√", "\\sqrt")
        is_correct = check_is_correct(fixed_answer, answer)
        final_answer = fixed_answer

    probs = F.softmax(output_scores, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    total_entropy = entropy.sum()
    pe = (total_entropy / len(entropy)).item()

    print(f"\n  Sample {sample_idx}: ")
    # print(f"    Answer text: {answer_text}")
    print(f"    Final answer: {final_answer}")
    print(f"    Is correct: {is_correct}")
    print(f"    Score: {pe}")

    paths.append({
        "prompt": question,
        "ground_truth": answer,
        "final_answer": final_answer,
        "is_correct": is_correct,
        "generated_ids": generated_ids,
        "prompt_len": prompt_len,
        "answer_ids": answer_ids,
        "answer_text": answer_text,
        "output_scores": output_scores,
        "score": pe 
    })


