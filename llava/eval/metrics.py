import json
import re
from typing import List, Dict


def extract_result(text: str) -> List[str]:
    """
    Extracts answers using a regex to find answers in the text
    """
    # Regular expression pattern to capture answer prefixes and extract answer characters
    answer_prefix_pattern = r"(Answer\(s\)?|Answers?)[:\s]*([A-D\d\(\), ]*)"
    answer_text = " ".join(
        match[1].strip().rstrip('.') for match in re.findall(answer_prefix_pattern, text, re.IGNORECASE))

    # Extract capital letters A-D or numbers in parentheses, remove duplicates, and sort
    results = sorted(set(re.findall(r"[A-D]|\(\d+\)", answer_text)))
    return results


class Scorer:
    """Class for computing evaluation metrics for model predictions."""

    def __init__(self):
        pass

    def _compute_subset_accuracy(self, preds: Dict[str, List[str]], true_answers: Dict[str, List[str]]) -> float:
        """
        Computes the Exam Score (Subset Accuracy): the proportion of questions
        where the model predicted all correct answers and no incorrect answers.
        """
        exact_matches = sum(set(pred) == set(true_answers[qid]) for qid, pred in preds.items())
        return 100 * exact_matches / len(preds) if preds else 0

    def _compute_precision_recall_f1(self, preds: Dict[str, List[str]], true_answers: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Computes Precision, Recall, and F1-score for multi-label classification.
        """
        true_positives, false_positives, false_negatives = 0, 0, 0

        for qid, pred in preds.items():
            true_set = set(true_answers[qid])
            pred_set = set(pred)

            true_positives += len(true_set & pred_set)       # Correctly predicted answers
            false_positives += len(pred_set - true_set)      # Incorrectly predicted answers
            false_negatives += len(true_set - pred_set)      # Missed correct answers

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision * 100, "recall": recall * 100, "f1_score": f1_score * 100}

    def compute_score(self, pred_file: str, true_answers_file: str) -> Dict[str, float]:
        """
        Computes Exam Score, Precision, Recall, and F1-score.
        """
        # Load predictions and true answers
        with open(pred_file, 'r') as f:
            preds = json.load(f)
        with open(true_answers_file, 'r') as f:
            true_answers = json.load(f)

        # Compute scores
        subset_accuracy = self._compute_subset_accuracy(preds, true_answers)
        precision_recall_f1 = self._compute_precision_recall_f1(preds, true_answers)

        # Display results
        print("=" * 50)
        print(f"Exam Score:\t {subset_accuracy:.2f}%")
        print(f"Precision:\t {precision_recall_f1['precision']:.2f}%")
        print(f"Recall:\t\t {precision_recall_f1['recall']:.2f}%")
        print(f"F1-Score:\t {precision_recall_f1['f1_score']:.2f}%")
        print("=" * 50)

        return {"exam_score": subset_accuracy, **precision_recall_f1}
