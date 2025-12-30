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

# for conf, alphas in quantile_conf_alpha_map.items():
#     for alpha in alphas:
#         name = f"quantile_{int(alpha*100)}_{conf}"
#         sampling_configs[name] = {
#             "method": name,
#             "scoring_mode": 'quantile',
#             "sampling_mode": "temperature",
#             "confidence": conf,
#             "alpha": alpha
#         }

# List of different settings to run
sampling_configs.update({

    "oracle": {
        "method": "oracle",
        "scoring_mode": '',
        "sampling_mode": "temperature",
        "confidence": "",
    },
    
    # "group_entropy": {
    #     "method": "group_entropy",
    #     "scoring_mode": 'mean',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    #     "k": 3
    # },

    # "gibbs_entropy_lin": {
    #     "method": "xentropy",
    #     "scoring_mode": 'mean',
    #     "sampling_mode": "temperature",
    #     "confidence": "gibbs_entropy_lin",
    # },

    # "gibbs_entropy_exp": {
    #     "method": "xentropy",
    #     "scoring_mode": 'mean',
    #     "sampling_mode": "temperature",
    #     "confidence": "gibbs_entropy_exp",
    # },

    # "tsallis_entropy_lin": {
    #     "method": "xentropy",
    #     "scoring_mode": 'mean',
    #     "sampling_mode": "temperature",
    #     "confidence": "tsallis_entropy_lin",
    #     "alpha": 0.4
    # },

    # "tsallis_entropy_exp": {
    #     "method": "xentropy",
    #     "scoring_mode": 'mean',
    #     "sampling_mode": "temperature",
    #     "confidence": "tsallis_entropy_exp",
    #     "alpha": 0.4
    # },

    # "renyi_entropy_lin": {
    #     "method": "xentropy",
    #     "scoring_mode": 'mean',
    #     "sampling_mode": "temperature",
    #     "confidence": "renyi_entropy_lin",
    #     "alpha": 0.4
    # },

    # "renyi_entropy_exp": {
    #     "method": "xentropy",
    #     "scoring_mode": 'mean',
    #     "sampling_mode": "temperature",
    #     "confidence": "renyi_entropy_exp",
    #     "alpha": 0.4
    # },

    # "trend_estimation": {
    #     "method": "trend_estimation",
    #     "estimation_method": "linear_regression",
    #     "scoring_mode": '',
    #     "sampling_mode": "temperature",
    #     "confidence": "",
    # },

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

    "energy_mean": {
        "method": "energy",
        "sampling_mode": "temperature",
        "temperature": 1.0,
        "mode": "mean"
    },

    "energy_worst": {
        "method": "energy",
        "sampling_mode": "temperature",
        "temperature": 1.0,
        "mode": "worst",
        "worst_ratio": 0.2
    },

    "energy_bottom_group": {
        "method": "energy",
        "sampling_mode": "temperature",
        "temperature": 1.0,
        "mode": "bottom_group",
        "window_size": 50,
        "top_k": 20,
        "bottom_percent": 0.1
    },

    "energy_lowest_group": {
        "method": "energy",
        "sampling_mode": "temperature",
        "temperature": 1.0,
        "mode": "lowest_group",
        "window_size": 50,
        "top_k": 20,
    },

    "energy_tail": {
        "method": "energy",
        "sampling_mode": "temperature",
        "temperature": 1.0,
        "mode": "tail",
        "tail_count": 50
    },

    "logit_mean": {
        "method": "logit",
        "sampling_mode": "temperature",
        "scoring_mode": 'mean',
    },

    # "logit_max": {
    #     "method": "logit",
    #     "sampling_mode": "temperature",
    #     "scoring_mode": 'max',
    # },

    # "logit_min": {
    #     "method": "logit",
    #     "sampling_mode": "temperature",
    #     "scoring_mode": 'min',
    # },

    # "logit_gap": {
    #     "method": "logit",
    #     "sampling_mode": "temperature",
    #     "scoring_mode": 'gap',
    # },

    # "logtoku": {
    #     "method": "logtoku",
    #     "sampling_mode": "temperature",
    #     "k_logits": 5,
    #     "k_worst_tokens": 25
    # },

    "deepconf_bottom_group_10": {
        "method": "deepconf",
        "sampling_mode": "temperature",
        "mode": "bottom_group",
        "window_size": 50,
        "top_k": 20,
        "bottom_percent": 0.1
    },

    "deepconf_lowest_group": {
        "method": "deepconf",
        "sampling_mode": "temperature",
        "mode": "lowest_group",
        "window_size": 50,
        "top_k": 20,
    },

    "deepconf_tail_50": {
        "method": "deepconf",
        "sampling_mode": "temperature",
        "mode": "tail",
        "tail_count": 50,
        "window_size": 50,
        "top_k": 20,
    },

    # "predictive_entropy": {
    #     "decoding_mode": '',
    #     "method": "pe",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "default",
    # },

    "normalized_likelihood": {
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

    # "topk_entropy": {
    #     "decoding_mode": '',
    #     "method": "topk_entropy",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "default",
    # },

    # "window_entropy": {
    #     "decoding_mode": '',
    #     "method": "window_entropy",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "default",
    # },

    # "stability_aware_entropy": {
    #     "decoding_mode": '',
    #     "method": "stability_aware_entropy",
    #     "scoring_mode": 'log',
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    #     "window_size": 10,
    #     "alpha": 0.5,
    #     "beta": 0.2
    # },

    "heterogeneous_ensemble_rrf": {
        "method": "rrf",
        "sampling_mode": "temperature",
        "rank_mode": "rrf",
        "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
        "rrf_k": 40,
        "power_p": 1.4,
        "weighting_mode": "std_inverse",
    },

    "heterogeneous_ensemble_borda": {
        "method": "borda",
        "sampling_mode": "temperature",
        "rank_mode": "borda",
        "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
        "rrf_k": 40,
        "power_p": 1.4,
        "weighting_mode": "std_inverse",
    },

    "heterogeneous_ensemble_combsum": {
        "method": "combsum",
        "sampling_mode": "temperature",
        "rank_mode": "combsum",
        "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
        "rrf_k": 5,
        "power_p": 0.3,
        "weighting_mode": "std_inverse",
    },

    # "heterogeneous_ensemble_combmnz": {
    #     "method": "combmnz",
    #     "sampling_mode": "temperature",
    #     "rank_mode": "combmnz",
    #     "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
    #     "rrf_k": 5,
    #     "power_p": 0.3,
    #     "weighting_mode": "std_inverse",
    # },

    "heterogeneous_ensemble_expcombsum": {
        "method": "expcombsum",
        "sampling_mode": "temperature",
        "rank_mode": "expcombsum",
        "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
        "rrf_k": 5,
        "power_p": 0.3,
        "weighting_mode": "std_inverse",
    },

    # "heterogeneous_ensemble_expcombmnz": {
    #     "method": "expcombmnz",
    #     "sampling_mode": "temperature",
    #     "rank_mode": "expcombmnz",
    #     "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
    #     "rrf_k": 5,
    #     "power_p": 0.3,
    #     "weighting_mode": "std_inverse",
    # },

    # "heterogeneous_ensemble_combsum_topn": {
    #     "method": "combsum_topn",
    #     "sampling_mode": "temperature",
    #     "rank_mode": "combsum_topn",
    #     "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
    #     "rrf_k": 5,
    #     "power_p": 0.3,
    #     "weighting_mode": "std_inverse",
    #     "top_n": 3,
    # },

    "heterogeneous_ensemble_combmax": {
        "method": "combmax",
        "sampling_mode": "temperature",
        "rank_mode": "combmax",
        "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
        "rrf_k": 5,
        "power_p": 0.3,
        "weighting_mode": "std_inverse",
        "top_n": 3,
    },

    "heterogeneous_ensemble_combmin": {
        "method": "combmin",
        "sampling_mode": "temperature",
        "rank_mode": "combmin",
        "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
        "rrf_k": 5,
        "power_p": 0.3,
        "weighting_mode": "std_inverse",
        "top_n": 3,
    },

    "heterogeneous_ensemble_combmed": {
        "method": "combmed",
        "sampling_mode": "temperature",
        "rank_mode": "combmed",
        "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
        "rrf_k": 5,
        "power_p": 0.3,
        "weighting_mode": "std_inverse",
        "top_n": 3,
    },

    "heterogeneous_ensemble_combanz": {
        "method": "combanz",
        "sampling_mode": "temperature",
        "rank_mode": "combanz",
        "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
        "rrf_k": 5,
        "power_p": 0.3,
        "weighting_mode": "std_inverse",
        "top_n": 3,
    },

    "heterogeneous_ensemble_sqcombsum": {
        "method": "sqcombsum",
        "sampling_mode": "temperature",
        "rank_mode": "sqcombsum",
        "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
        "rrf_k": 5,
        "power_p": 0.3,
        "weighting_mode": "std_inverse",
        "top_n": 3,
    },

    # "heterogeneous_ensemble_sqcombmnz": {
    #     "method": "sqcombmnz",
    #     "sampling_mode": "temperature",
    #     "rank_mode": "sqcombmnz",
    #     "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
    #     "rrf_k": 5,
    #     "power_p": 0.3,
    #     "weighting_mode": "std_inverse",
    #     "top_n": 3,
    # },

    # "heterogeneous_ensemble_rrx": {
    #     "method": "rr_x",
    #     "sampling_mode": "temperature",
    #     "rank_mode": "rr_x",
    #     "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
    #     "rrf_k": 5,
    #     "power_p": 0.7,
    #     "weighting_mode": "std_inverse",
    #     "top_n": 3,
    # },

    # "heterogeneous_ensemble_combsum_rrx": {
    #     "method": "combsum_rr_x",
    #     "sampling_mode": "temperature",
    #     "rank_mode": "combsum_rr_x",
    #     "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
    #     "rrf_k": 5,
    #     "power_p": 0.7,
    #     "weighting_mode": "std_inverse",
    #     "top_n": 3,
    # },

    # "heterogeneous_ensemble_sqcombsum_rrx": {
    #     "method": "sqcombsum_rr_x",
    #     "sampling_mode": "temperature",
    #     "rank_mode": "sqcombsum_rr_x",
    #     "ensemble_metrics": ["mean_entropy", "quantile_entropy_5", "quantile_entropy_10", "quantile_entropy_15", "quantile_entropy_20", "quantile_entropy_25", "quantile_entropy_30", "quantile_entropy_35", "quantile_entropy_40", "quantile_entropy_45", "quantile_entropy_50", "quantile_entropy_55", "quantile_entropy_60", "quantile_entropy_65", "quantile_entropy_70", "quantile_entropy_75", "quantile_entropy_80", "quantile_entropy_85", "quantile_entropy_90", "quantile_entropy_95"],
    #     "rrf_k": 5,
    #     "power_p": 0.7,
    #     "weighting_mode": "std_inverse",
    #     "top_n": 3,
    # },

    # "distinct_entropy": {
    #     "method": "distinct_entropy",
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    #     "distinct_ngram_n": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     "distinct_threshold_ratio": 0.6
    # },

    # "permutation_entropy": {
    #     "method": "permutation_entropy",
    #     "sampling_mode": "temperature",
    #     "confidence": "entropy",
    # },

    "self_certainty": {
        "method": "self_certainty",
        "sampling_mode": "temperature",
        "confidence": "kl",
        "borda_p": 0.3,
    },

    # "stable_rank": {
    #     "method": "stable_rank",
    #     "sampling_mode": "temperature",
    # },

    # "rank_weighted_confidence": {
    #     "method": "rank_weighted_confidence",
    #     "sampling_mode": "temperature",
    # }

    # "generalized_entropy": {
    #     "method": "generalized_entropy",
    #     "sampling_mode": "temperature",
    #     "gamma": 0.5,
    #     "top_m": 2,
    # },

    # "trajectory_pattern": {
    #     "method": "trajectory_pattern",
    #     "sampling_mode": "temperature",
    # },

    # "anisotropy_evolution": {
    #     "method": "anisotropy_evolution",
    #     "sampling_mode": "temperature",
    # },

    # "hidden_complexity": {
    #     "method": "hidden_complexity",
    #     "sampling_mode": "temperature",
    # },

    # "attention_entropy": {
    #     "method": "attention_entropy",
    #     "sampling_mode": "temperature",
    #     "scoring_mode": "mean",
    #     "top_k": 5,
    # },

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

# sampling_configs.update({

#     "cer_prob_product_log_last": {
#         "decoding_mode": 'last',
#         "method": "cer",
#         "scoring_mode": 'log',
#         "sampling_mode": "temperature",
#         "confidence": "product",
#     },

#     "cer_entropy_log_last": {
#         "decoding_mode": 'last',
#         "method": "cer",
#         "scoring_mode": 'log',
#         "sampling_mode": "temperature",
#         "confidence": "entropy",
#     },

# })

# All the methods to be evaluated
method_groups = {
    # "greedy": greedy_configs,
    "sampling": sampling_configs,
}

# General configuration
@dataclass
class Config:
    # Path to the HuggingFace model or local directory
    # /models/Qwen/Qwen2.5-7B-Instruct
    # /models/meta-llama/Llama-3.1-8B-Instruct
    model_name: str = os.getenv(
        "MODEL_NAME", "/models/Qwen/Qwen2.5-7B-Instruct"
    )
    lingua_model_name: str = os.getenv(
        "LLMLINGUA_MODEL_NAME", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    )
    model_type: str = os.getenv("MODEL_TYPE", "qwen")
    
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
    top_p: float = float(os.getenv("TOP_P", 0.95))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", 1.0))
    length_penalty: float = float(os.getenv("LENGTH_PENALTY", 1.0))
    no_repeat_ngram_size: int = int(os.getenv("NO_REPEAT_NGRAM_SIZE", 0))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", 4096))
    early_stopping: bool = False
    
    # True: aggregate paths, False: the best path
    aggregate: bool = True
    
    # Number of samples to process
    # number_samples: int = int(os.getenv("N_SAMPLE", 500))
    seed: int = int(os.getenv("SEED", 1252))

    # Path to few-shots
    gsm8k_shots: str = "inputs/shots/gsm8k.txt"
    allenai_shots: str = "inputs/shots/allenai.txt"
    math_shots: str = "inputs/shots/math.txt"

    datasets = eval(os.getenv("DATASETS", """{
        "gsm8k_hard": "gsmhardv2.jsonl",
        "math_500": "math_500.jsonl",
        "math": "math.jsonl",
        "gsm8k": "gsm8k.jsonl",
        "aime_2025": "aime_2025.jsonl",
        "brumo_2025": "brumo_2025.jsonl",
    }"""))

    # For test
    datasets = eval(os.getenv("DATASETS", """{
        "math": "math.jsonl",
    }"""))

    batch_size = int(os.getenv("BATCH_SIZE", 1))

    verbose: bool = eval(os.getenv("VERBOSE", 'False'))

    use_base_prompt: bool = eval(os.getenv("BASE_PROMPT", 'True'))

    exclude_gpus: str = "0, 1, 2, 3, 4, 5, 6"
