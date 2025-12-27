def self_consistency(sample_paths):
    
    voting_results = dict()

    for path in sample_paths:
        final_answer = path["final_answer"]
        if final_answer not in voting_results:
            voting_results[final_answer] = 1
        else:
            voting_results[final_answer] += 1

    best_answer = max(voting_results, key=voting_results.get)
    confidence = voting_results[best_answer] / len(sample_paths)

    answer_text = None
    for path in sample_paths:
        if path["final_answer"] == best_answer:
            answer_text = path["answer_text"]
            break

    path_info = [
        {
            "answer_text": path["answer_text"],
            "score": voting_results[path["final_answer"]] / len(sample_paths),
            "final_answer": path["final_answer"]
        }
        for path in sample_paths
    ]

    return answer_text, confidence, best_answer, path_info


