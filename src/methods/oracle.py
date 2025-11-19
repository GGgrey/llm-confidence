from src.grader import check_is_correct


def oracle_self_consistency(sample_paths):

    correct_paths = []
    ground_truth = sample_paths[0].get("ground_truth", None)

    for path in sample_paths:
        final_answer = path["final_answer"]
        if check_is_correct(final_answer, ground_truth):
            correct_paths.append(path)

    if len(correct_paths) > 0:
        oracle_path = correct_paths[0]
        oracle_answer = oracle_path["final_answer"]
        oracle_confidence = len(correct_paths) / len(sample_paths)
        oracle_text = oracle_path.get("answer_text", "")
    else:
        oracle_answer = ""
        oracle_confidence = 0.0
        oracle_text = ""

    return oracle_text, oracle_confidence, oracle_answer
