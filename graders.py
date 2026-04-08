# graders.py
"""
Graders for Red-Blue Adversarial Federated Learning Environment
3 tasks with increasing difficulty
"""

from typing import Dict, Any, List

def grade_task1_recall(episode_history: List[Dict], num_rounds: int) -> float:
    """Task 1: Detection Recall (Easy)"""
    total_detections = sum(int(h.get("Correct Detections", 0)) for h in episode_history)
    total_fns = sum(int(h.get("False Negatives", 0)) for h in episode_history)
    recall = total_detections / (total_detections + total_fns) if (total_detections + total_fns) > 0 else 0
    return _clamp(min(1.0, recall / 0.8))


def grade_task2_precision(episode_history: List[Dict], num_rounds: int) -> float:
    """Task 2: Precision & Pattern Recognition (Medium)"""
    total_detections = sum(int(h.get("Correct Detections", 0)) for h in episode_history)
    total_fps = sum(int(h.get("False Positives", 0)) for h in episode_history)
    precision = total_detections / (total_detections + total_fps) if (total_detections + total_fps) > 0 else 0
    return _clamp(min(1.0, precision / 0.75)) if total_detections > 0 else 0.0


def grade_task3_resilience(episode_history: List[Dict], num_rounds: int) -> float:
    """Task 3: Adversarial Resilience & Efficiency (Hard)"""
    advanced_attacks = ["coordinated", "alie", "imitation"]
    advanced_detections = 0
    first_detection_round = num_rounds + 1
    total_detections = sum(int(h.get("Correct Detections", 0)) for h in episode_history)
    total_fns = sum(int(h.get("False Negatives", 0)) for h in episode_history)

    for h in episode_history:
        atk_type = h.get("Attacker Action", "").split(" ")[0].lower()
        correct = int(h.get("Correct Detections", 0))
        if correct > 0:
            if atk_type in advanced_attacks:
                advanced_detections += 1
            if h.get("Round", 99) < first_detection_round:
                first_detection_round = h.get("Round", 99)

    recall = total_detections / (total_detections + total_fns) if (total_detections + total_fns) > 0 else 0
    efficiency_bonus = max(0, (5 - first_detection_round) / 4) if first_detection_round <= num_rounds else 0
    task3_score = (min(1.0, advanced_detections / 3) * 0.5) + (recall * 0.3) + (efficiency_bonus * 0.2)
    return _clamp(min(1.0, task3_score))


def _clamp(score: float) -> float:
    """Snap scores very close to 0 or 1 to exactly 0 or 1, otherwise clamp to [0.01, 0.99]."""
    if score < 0.01:
        return 0.0
    if score > 0.99:
        return 1.0
    return round(score, 2)

TASK_DEFINITIONS = [
    {
        "id": "task1",
        "name": "Task 1 (Easy - Detection Recall)",
        "difficulty": "easy",
        "description": "Measure defender recall of malicious client detection."
    },
    {
        "id": "task2",
        "name": "Task 2 (Medium - Pattern Precision)",
        "difficulty": "medium",
        "description": "Measure defender precision and pattern recognition."
    },
    {
        "id": "task3",
        "name": "Task 3 (Hard - Adversarial Resilience & Speed)",
        "difficulty": "hard",
        "description": "Measure advanced adversarial resilience and detection speed."
    }
]

# Grader metadata exposing all 3 graders for validator discovery
GRADER_DEFINITIONS = [
    {
        "id": "task1",
        "task_id": "task1",
        "name": "Detection Recall Grader",
        "difficulty": "easy",
        "description": "Evaluates defender recall of malicious client detection.",
        "grader_function": "grade_task1_recall"
    },
    {
        "id": "task2",
        "task_id": "task2",
        "name": "Pattern Precision Grader",
        "difficulty": "medium",
        "description": "Evaluates defender precision and pattern recognition.",
        "grader_function": "grade_task2_precision"
    },
    {
        "id": "task3",
        "task_id": "task3",
        "name": "Adversarial Resilience Grader",
        "difficulty": "hard",
        "description": "Evaluates advanced adversarial resilience and detection speed.",
        "grader_function": "grade_task3_resilience"
    }
]

# Required by validator for grader discovery — maps task id → grader function





def run_all_graders(episode_history: List[Dict], num_rounds: int, difficulty: str = "medium") -> Dict:
    """
    Compute 3 graded tasks for the episode based on agent performance.
    Returns scores between 0.0 and 1.0 for each task.
    """
    return {
        "task1_recall": grade_task1_recall(episode_history, num_rounds),
        "task2_precision": grade_task2_precision(episode_history, num_rounds),
        "task3_hard_mode": grade_task3_resilience(episode_history, num_rounds),
        "overall_balance": _clamp(
            0.5 + 0.5 * (
                sum(float(h.get("D_Reward", 0)) for h in episode_history) /
                (sum(float(h.get("A_Reward", 0)) for h in episode_history) + 1e-6)
            )
        ),
        "total_detections": sum(int(h.get("Correct Detections", 0)) for h in episode_history),
        "total_fns": sum(int(h.get("False Negatives", 0)) for h in episode_history),
        "total_fps": sum(int(h.get("False Positives", 0)) for h in episode_history),
        "first_detection_round": min((h.get("Round", num_rounds + 1) for h in episode_history), default=num_rounds + 1)
    }

def grader_summary(episode_history: List[Dict], num_rounds: int, difficulty: str = "medium") -> Dict:
    """Return formatted summary for display"""
    scores = run_all_graders(episode_history, num_rounds, difficulty)
    
    return {
        "status": "completed",
        "tasks": {
            "Task 1 (Easy - Detection Recall)": scores['task1_recall'],
            "Task 2 (Medium - Pattern Precision)": scores['task2_precision'],
            "Task 3 (Hard - Adversarial Resilience & Speed)": scores['task3_hard_mode']
        },
        "overall": {
            "Total True Positives": scores['total_detections'],
            "Total False Negatives": scores['total_fns'],
            "Total False Positives": scores['total_fps'],
            "Speed (First Detect)": f"Round {scores['first_detection_round']}",
            "Balance Score": scores['overall_balance']
        }
    }
GRADERS = {
    "task1": grade_task1_recall,
    "task2": grade_task2_precision,
    "task3": grade_task3_resilience,
}