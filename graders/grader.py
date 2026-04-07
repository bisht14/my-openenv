# graders/grader.py

from typing import List, Tuple


def _normalize(text: str) -> str:
    return text.lower().strip()


def _precision_recall(pred: List[str], truth: List[str]) -> Tuple[float, float]:
    pred = [_normalize(p) for p in pred]
    truth = [_normalize(t) for t in truth]

    tp = sum(1 for p in pred if any(t in p or p in t for t in truth))
    fp = len(pred) - tp
    fn = len(truth) - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return precision, recall


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _severity_score(pred: List[str], truth: List[str]) -> float:

    if pred is None or truth is None:
        return 0.0
    
    mapping = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    score = 0
    for p, t in zip(pred, truth):
        if p == t:
            score += 1
        elif abs(mapping.get(p, 0) - mapping.get(t, 0)) == 1:
            score += 0.5

    return score / max(len(truth), 1)


def _explanation_score(text: str) -> float:
    if not text:
        return 0.0

    length_score = min(len(text) / 200, 1.0)
    fix_bonus = 0.2 if ("fix" in text.lower() or "use" in text.lower()) else 0

    return min(length_score + fix_bonus, 1.0)


def grade(action, spec):
    truth_issues = spec.ground_truth_issues
    # truth_severity = spec.ground_truth_severity
    truth_severity = ["medium"] * len(spec.ground_truth_issues)

    pred_issues = action.issues_found
    pred_severity = action.severity_tags

    precision, recall = _precision_recall(pred_issues, truth_issues)
    f1_score = _f1(precision, recall)

    severity = _severity_score(pred_severity, truth_severity)
    explanation = _explanation_score(action.explanation)

    # 🚨 penalty for hallucinations
    hallucination_penalty = max(0, len(pred_issues) - len(truth_issues)) * 0.05

    total = (
        0.5 * f1_score +
        0.2 * severity +
        0.3 * explanation
    )

    total = max(0.0, min(1.0, total - hallucination_penalty))

    return total, {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "issue_score": f1_score,
        "severity_score": severity,
        "explanation_score": explanation,
        "total_reward": total,
    }