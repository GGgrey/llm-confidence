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
        "method": "greedy",
        "sampling_mode": "greedy",
    },
}

# Quantile-based methods
quantile_conf_alpha_map = {
    "prob": [x / 100 for x in range(0, 101, 5)],
    "entropy": [x / 100 for x in range(0, 101, 5)],
    "logit": [x / 100 for x in range(0, 101, 5)],
}
sampling_configs = {}

for conf, alphas in quantile_conf_alpha_map.items():
    for alpha in alphas:
        name = f"quantile_{int(alpha*100)}_{conf}"
        sampling_configs[name] = {
            "method": name,
            "scoring_mode": 'quantile',
            "sampling_mode": "temperature",
            "confidence": conf,
            "alpha": alpha
        }

# List of different settings to run
sampling_configs.update({

    "oracle": {
        "method": "oracle",
        "scoring_mode": '',
        "sampling_mode": "temperature",
        "confidence": "",
    },
    
    "group_entropy": {
        "method": "group_entropy",
        "scoring_mode": 'mean',
        "sampling_mode": "temperature",
        "confidence": "entropy",
        "k": 3
    },

    "gibbs_entropy_lin": {
        "method": "xentropy",
        "scoring_mode": 'mean',
        "sampling_mode": "temperature",
        "confidence": "gibbs_entropy_lin",
    },

    "gibbs_entropy_exp": {
        "method": "xentropy",
        "scoring_mode": 'mean',
        "sampling_mode": "temperature",
        "confidence": "gibbs_entropy_exp",
    },

    "tsallis_entropy_lin": {
        "method": "xentropy",
        "scoring_mode": 'mean',
        "sampling_mode": "temperature",
        "confidence": "tsallis_entropy_lin",
        "alpha": 0.4
    },

    "tsallis_entropy_exp": {
        "method": "xentropy",
        "scoring_mode": 'mean',
        "sampling_mode": "temperature",
        "confidence": "tsallis_entropy_exp",
        "alpha": 0.4
    },

    "renyi_entropy_lin": {
        "method": "xentropy",
        "scoring_mode": 'mean',
        "sampling_mode": "temperature",
        "confidence": "renyi_entropy_lin",
        "alpha": 0.4
    },

    "renyi_entropy_exp": {
        "method": "xentropy",
        "scoring_mode": 'mean',
        "sampling_mode": "temperature",
        "confidence": "renyi_entropy_exp",
        "alpha": 0.4
    },

    "trend_estimation": {
        "method": "trend_estimation",
        "estimation_method": "linear_regression",
        "scoring_mode": '',
        "sampling_mode": "temperature",
        "confidence": "",
    },

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
        "confidence": "product"
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
        "confidence": "product"
    },

    "cer_prob_product_max_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'max',
        "sampling_mode": "temperature",
        "confidence": "product"
    },

    "cer_prob_product_h_mean_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'h_mean',
        "sampling_mode": "temperature",
        "confidence": "product"
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

    "cer_prob_product_weighted_mean_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'weighted_mean',
        "sampling_mode": "temperature",
        "confidence": "product",
    },

    "cer_prob_product_weighted_half_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'weighted_half',
        "sampling_mode": "temperature",
        "confidence": "product",
    },

    "cer_prob_product_weighted_2_all": {
        "decoding_mode": 'all',
        "method": "cer",
        "scoring_mode": 'weighted_2',
        "sampling_mode": "temperature",
        "confidence": "product",
    },

    "cer_prob_product_log_last": {
        "decoding_mode": 'last',
        "method": "cer",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "product",
    },

    "cer_entropy_log_last": {
        "decoding_mode": 'last',
        "method": "cer",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "entropy",
    },

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

    "predictive_entropy": {
        "decoding_mode": '',
        "method": "pe",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    "normalized_likelihood": {
        "decoding_mode": '',
        "method": "nl",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    "likelihood": {
        "decoding_mode": '',
        "method": "ll",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    "normalized_entropy": {
        "decoding_mode": '',
        "method": "ne",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

    "perplexity": {
        "decoding_mode": '',
        "method": "ppl",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "default",
    },

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

    "stability_aware_entropy": {
        "decoding_mode": '',
        "method": "stability_aware_entropy",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "entropy",
        "window_size": 10,
        "alpha": 0.5,
        "beta": 0.2
    },

    "heterogeneous_ensemble_rrf": {
        "method": "stability_aware_entropy",
        "sampling_mode": "temperature",
        "rank_mode": "rrf",
        "ensemble_metrics": ["mean_entropy", "mean_logprob", "min_prob", "quantile_logit_0", "quantile_logit_5", "quantile_logit_10", "quantile_logit_15", "quantile_logit_20", "quantile_logit_25", "quantile_logit_30", "quantile_logit_35", "quantile_logit_40", "quantile_logit_45", "quantile_logit_50", "quantile_logit_55", "quantile_logit_60", "quantile_logit_65", "quantile_logit_70", "quantile_logit_75", "quantile_logit_80", "quantile_logit_85", "quantile_logit_90", "quantile_logit_95", "quantile_logit_100"],
        "rrf_k": 10,
        "weighting_mode": "std_inverse",
    },

    "semantic_consistency": {
        "decoding_mode": '',
        "method": "stability_aware_entropy",
        "scoring_mode": 'log',
        "sampling_mode": "temperature",
        "confidence": "entropy",
    },

    # "attention_dynamic": {
    #     "method": "attention_dynamic",
    #     "scoring_mode": 'union',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    # },

    # "attention_weighted_confidence": {
    #     "method": "attention_weighted_confidence",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    # },

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

})

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
        "MODEL_NAME", "/models/meta-llama/Llama-3.1-8B-Instruct"
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
    temperature: float = float(os.getenv("TEMPERATURE", 0.7))
    top_p: float = float(os.getenv("TOP_P", 0.95))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", 1.0))
    length_penalty: float = float(os.getenv("LENGTH_PENALTY", 1.0))
    no_repeat_ngram_size: int = int(os.getenv("NO_REPEAT_NGRAM_SIZE", 0))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", 2048))
    early_stopping: bool = False
    
    # True: aggregate paths, False: the best path
    aggregate: bool = True
    
    # Number of samples to process
    number_samples: int = int(os.getenv("N_SAMPLE", 500))
    seed: int = int(os.getenv("SEED", 101))

    # Path to few-shots
    gsm8k_shots: str = "inputs/shots/gsm8k.txt"
    allenai_shots: str = "inputs/shots/allenai.txt"
    math_shots: str = "inputs/shots/math.txt"

    datasets = eval(os.getenv("DATASETS", """{
        "gsm8k_hard": "gsmhardv2.jsonl",
        "math_500": "math_500.jsonl",
        "gsm8k": "gsm8k.jsonl",
        "aime_2025": "aime_2025.jsonl",
        "brumo_2025": "brumo_2025.jsonl",
    }"""))

    # For test
    datasets = eval(os.getenv("DATASETS", """{
        "math_500": "math_500.jsonl",
    }"""))

    batch_size = int(os.getenv("BATCH_SIZE", 1))

    verbose: bool = eval(os.getenv("VERBOSE", 'False'))

    use_base_prompt: bool = eval(os.getenv("BASE_PROMPT", 'True'))

    exclude_gpus: str = "0, 2, 3, 4, 5, 6, 7"
