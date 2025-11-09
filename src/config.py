from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv(override=True)  # Reads .env file and loads environment variables

# CER configs
cer_configs = {
    "cer_prob_sum_log_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "sum"
    },

    "cer_prob_product_log_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default"
    },

    "cer_entropy_log_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "entropy"
    },

    "cer_prob_product_min_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'min',
        "sampling_mode": "temperature",
        "confidence": "default"
    },

    "cer_prob_product_max_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'max',
        "sampling_mode": "temperature",
        "confidence": "default"
    },

    "cer_prob_product_h_mean_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'h_mean',
        "sampling_mode": "temperature",
        "confidence": "default"
    },

    "cer_entropy_weighted_mean_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'weighted_mean',
        "sampling_mode": "temperature",
        "confidence": "entropy"
    },

    "cer_entropy_min_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'min',
        "sampling_mode": "temperature",
        "confidence": "entropy"
    },

    "cer_entropy_h_mean_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'h_mean',
        "sampling_mode": "temperature",
        "confidence": "entropy"
    },

    "cer_entropy_max_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'max',
        "sampling_mode": "temperature",
        "confidence": "entropy"
    },

    "cer_prob_product_log_last": {
        "decoding_mode": 'last',
        "method": "cer",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    "cer_entropy_log_last": {
        "decoding_mode": 'last',
        "method": "cer",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "entropy",
    },

    "cer_prob_product_weighted_mean_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'weighted_mean',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    "cer_prob_product_weighted_half_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'weighted_half',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    "cer_prob_product_weighted_2_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'weighted_2',
        "sampling_mode": "temperature",
        "confidence": "default",
    },
}

# Greedy configs
greedy_configs = {
    "greedy": {
        "decoding_mode": 'all',
        "method": "greedy",
        "scoring_mode": 'log',
        "sampling_mode": "greedy",
        "confidence": "default",
    },
}

# List of different settings to run
sampling_configs = {
    
    "group_entropy": {
        "decoding_mode": '',
        "method": "group_entropy",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "entropy",
    },

    # "attention_dynamic": {
    #     "decoding_mode": '',
    #     "method": "attention_dynamic",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    # },

    # "trend_estimation": {
    #     "decoding_mode": '',
    #     "method": "trend_estimation",
    #     "estimation_method": "linear_regression",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    # },

    # "attention_weighted_confidence": {
    #     "decoding_mode": '',
    #     "method": "attention_weighted_confidence",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    # },

    "cer_entropy_weighted_mean_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'weighted_mean',
        "sampling_mode": "temperature",
        "confidence": "entropy"
    },

    "cer_prob_product_log_last": {
        "decoding_mode": 'last',
        "method": "cer",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    # "cer_entropy_log_last": {
    #     "decoding_mode": 'last',
    #     "method": "cer",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    # },

    "self_consistency": {
        "decoding_mode": '',
        "method": 'self_consistency',
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    "p_true": {
        "decoding_mode": '',
        "method": "p_true",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    # "predictive_entropy": {
    #     "decoding_mode": '',
    #     "method": "pe",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "default",
    # },

    "normilized_likelihood": {
        "decoding_mode": '',
        "method": "nl",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    # "likelihood": {
    #     "decoding_mode": '',
    #     "method": "ll",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "default",
    # },

    "normilized_entropy": {
        "decoding_mode": '',
        "method": "ne",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    # "perplexity": {
    #     "decoding_mode": '',
    #     "method": "ppl",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "default",
    # },

    "topk_entropy": {
        "decoding_mode": '',
        "method": "topk_entropy",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    "window_entropy": {
        "decoding_mode": '',
        "method": "window_entropy",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    # "key_confidence": {
    #     "decoding_mode": '',
    #     "method": "key_confidence",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    # },

    # "hidden_svd": {
    #     "decoding_mode": '',
    #     "method": "hidden_svd",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "default",
    # },

    # "attention_eigenvalue": {
    #     "decoding_mode": '',
    #     "method": "attn_eigen",
    #     "scoring_mode": 'top_k',
    #     "sampling_mode": "temperature",
    #     "confidence": "default",
    # },

    # "compression_confidence": {
    #     "decoding_mode": '',
    #     "method": "compression",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    #     "compression_ratio": 0.5
    # },
}

# All the methods to be evaluated
method_groups = {
    # "greedy": greedy_configs,
    "sampling": sampling_configs,
}

# General configuration
@dataclass
class Config:
    # Path to the HuggingFace model or local directory
    model_name: str = os.getenv(
        "MODEL_NAME", "/data/sunqiao/projects/models/Llama-3.1-8B-Instruct"
    )
    lingua_model_name: str = os.getenv(
        "LLMLINGUA_MODEL_NAME", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    )
    model_type: str = os.getenv("MODEL_TYPE", "llama")
    
    data_dir = os.getenv("DATA_DIR", "data")

    # Load the model from the local directory instead of the HF
    read_model_from_huggingface: bool = eval(os.getenv("LOCAL_MODEL", 'True'))

    # Huggingface token
    hugging_face_token: str = os.getenv(
        "HUGGING_FACE_TOKEN", "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"
    )

    # Sampling parameters
    k: int = int(os.getenv("K", 16))
    num_beams: int = int(os.getenv("NUM_BEAMS", 1))
    temperature: float = float(os.getenv("TEMPERATURE", 1.0))
    top_p: float = float(os.getenv("TOP_P", 1.0))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", 1.0))
    length_penalty: float = float(os.getenv("LENGTH_PENALTY", 1.0))
    no_repeat_ngram_size: int = int(os.getenv("NO_REPEAT_NGRAM_SIZE", 0))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", 2048))
    early_stopping: bool = False
    
    # True: aggregate paths, False: the best path
    aggregate: bool = True
    
    # Number of samples to process
    number_samples: int = int(os.getenv("N_SAMPLE", 100))
    seed: int = int(os.getenv("SEED", 102))

    # Path to few-shots
    gsm8k_shots: str = "inputs/shots/gsm8k.txt"
    allenai_shots: str = "inputs/shots/allenai.txt"
    math_shots: str = "inputs/shots/math.txt"

    datasets = eval(os.getenv("DATASETS", """{
        "allenai": "allenai_math_qa_test_processed.parquet",
        "math": "src_custom_datasets_math_dataset_test_processed.parquet",
        "gsm8k": "openai_gsm8k_test_processed.parquet",
    }"""))

    # For test
    datasets = eval(os.getenv("DATASETS", """{
        "math": "src_custom_datasets_math_dataset_test_processed.parquet",
    }"""))

    batch_size = int(os.getenv("BATCH_SIZE", 1))

    verbose: bool = eval(os.getenv("VERBOSE", 'False'))

    use_base_prompt: bool = eval(os.getenv("BASE_PROMPT", 'True'))
