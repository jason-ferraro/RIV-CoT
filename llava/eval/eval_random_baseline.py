import random
import argparse
import numpy as np
from itertools import combinations, product
from llava.eval.metrics import Scorer
from llava.riv_cot_utils import save_json
from typing import Dict, List



def calculate_exact_possible_score(questions_data: Dict[str, Dict]) -> float:
    """
    Calculate the exact possible exam score considering all possible subsets for each question.
    Handles multi-part questions by ensuring at least one correct answer per part.

    Parameters:
        questions_data (dict): Dictionary of question data containing possible answers and true answers.

    Returns:
        float: Exact possible exam score as a percentage.
    """
    total_score = 0.0

    for question_data in questions_data.values():
        possible_answers = question_data['possible_answers']
        true_answers = set(question_data['true_answers'])
        has_multiple_questions = question_data['has_multiple_questions']

        if has_multiple_questions:
            part_combinations = list(product(*[possible_answers.keys()]))
            correct_subsets = sum(any(ans in true_answers for ans in combo) for combo in part_combinations)
            total_combinations = len(part_combinations)
        else:
            all_possible_subsets = [set(subset) for i in range(1, len(possible_answers) + 1)
                                    for subset in combinations(possible_answers.keys(), i)]
            total_combinations = len(all_possible_subsets)
            correct_subsets = sum(1 for subset in all_possible_subsets if subset == true_answers)

        correct_probability = (correct_subsets / total_combinations) if total_combinations else 0
        total_score += correct_probability
    exact_possible_exam_score = (total_score / len(questions_data)) * 100

    return exact_possible_exam_score


def generate_random_predictions(questions_data: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Generate a single random prediction for each question, ensuring for multi-part questions
    that at least one correct answer per question part is selected.

    Parameters:
        questions_data (dict): Dictionary containing question-answer pairs with possible answers.

    Returns:
        dict: Random predictions for each question ID.
    """
    random_predictions = {}

    for question_id, question_data in questions_data.items():
        possible_answers = question_data['possible_answers']
        has_multiple_questions = question_data['has_multiple_questions']

        # Generate predictions based on whether the question has multiple parts
        if has_multiple_questions:
            # Separate answer options for each part and select at least one answer per part
            part_combinations = product(*[list(possible_answers.keys())])
            random_combination = random.choice([combo for combo in part_combinations if combo])
            random_predictions[question_id] = list(random_combination)
        else:
            # Single question case: generate all non-empty combinations and choose one at random
            options = list(possible_answers.keys())
            all_combinations = [combo for i in range(1, len(options) + 1) for combo in combinations(options, i)]
            random_combination = random.choice(all_combinations)
            random_predictions[question_id] = list(random_combination)

    return random_predictions


def main():
    parser = argparse.ArgumentParser(description="Calculate random baseline with scoring metrics for QA dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--num_trials", type=int, default=1000,
                        help="Number of trials for averaging random predictions")

    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_path, 'r') as file:
        questions_data = json.load(file)

    # Calculate exact possible exam score
    exact_possible_exam_score = calculate_exact_possible_score(questions_data)

    # Initialize scorer and result storage
    scorer = Scorer()
    exam_scores, precisions, recalls, f1_scores = [], [], [], []

    for _ in range(args.num_trials):
        # Generate random predictions
        random_predictions = generate_random_predictions(questions_data)

        # Extract true answers
        true_answers = {qid: data['true_answers'] for qid, data in questions_data.items()}

        # Save predictions and true answers
        save_json(random_predictions, 'random_predictions.json')
        save_json(true_answers, 'true_answers.json')

        # Compute scores
        scores = scorer.compute_score('random_predictions.json', 'true_answers.json')

        # Collect scores for statistics
        exam_scores.append(scores['exam_score'])
        precisions.append(scores['precision'])
        recalls.append(scores['recall'])
        f1_scores.append(scores['f1_score'])

    # Calculate mean and std deviation for each score
    score_summary = {
        "exam_score": {"mean": np.mean(exam_scores), "std": np.std(exam_scores)},
        "precision": {"mean": np.mean(precisions), "std": np.std(precisions)},
        "recall": {"mean": np.mean(recalls), "std": np.std(recalls)},
        "f1_score": {"mean": np.mean(f1_scores), "std": np.std(f1_scores)}
    }

    # Display score summary and exact possible score
    print("=" * 50)
    print(f"Exact Possible Exam Score: {exact_possible_exam_score:.2f}%")
    for score_type, stats in score_summary.items():
        print(f"{score_type.capitalize()} - Mean: {stats['mean']:.2f}%, Std Dev: {stats['std']:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()


