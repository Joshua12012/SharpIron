# graders.py
"""
Graders for Red-Blue Adversarial Federated Learning Environment
3 tasks with increasing difficulty
"""

from typing import Dict, Any, List

def run_all_graders(episode_history: List[Dict], num_rounds: int, difficulty: str = "medium") -> Dict:
    """
    Compute 3 graded tasks for the episode based on agent performance.
    Returns scores between 0.0 and 1.0 for each task.
    """
    difficulty = difficulty.lower()
    
    total_detections = sum(int(h.get("Correct Detections", 0)) for h in episode_history)
    total_fns = sum(int(h.get("False Negatives", 0)) for h in episode_history)
    total_fps = sum(int(h.get("False Positives", 0)) for h in episode_history)
    
    # Task 1: Detection Recall (Easy)
    # How many of the actual attacks did the defender catch?
    # Target: 0.8+ for 100% score
    recall = total_detections / (total_detections + total_fns) if (total_detections + total_fns) > 0 else 0
    task1_score = min(1.0, recall / 0.8)
    
    # Task 2: Precision & Pattern Recognition (Medium)
    # Measures F1 or Precision (avoiding random guessing)
    # 0.7+ Precision for 100% score
    precision = total_detections / (total_detections + total_fps) if (total_detections + total_fps) > 0 else 0
    task2_score = min(1.0, precision / 0.75) if total_detections > 0 else 0
    
    # Task 3: Adversarial Resilience & Efficiency (Hard)
    # Score specifically for non-single attacks AND detecting early
    advanced_attacks = ["coordinated", "alie", "imitation"]
    advanced_detections = 0
    first_detection_round = num_rounds + 1
    
    for h in episode_history:
        atk_type = h.get("Attacker Action", "").split(" ")[0].lower()
        correct = int(h.get("Correct Detections", 0))
        if correct > 0:
            if atk_type in advanced_attacks:
                advanced_detections += 1
            if h.get("Round", 99) < first_detection_round:
                first_detection_round = h.get("Round", 99)

    # Efficiency Bonus: Detect before round 4
    efficiency_bonus = max(0, (5 - first_detection_round) / 4) if first_detection_round <= num_rounds else 0
    
    # Base score on advanced detections (50%) + overall recall (30%) + efficiency (20%)
    task3_score = (min(1.0, advanced_detections / 3) * 0.5) + (recall * 0.3) + (efficiency_bonus * 0.2)
    task3_score = min(1.0, task3_score)
    
    # Overall balance check
    total_a_reward = sum(float(h.get("A_Reward", 0)) for h in episode_history)
    total_d_reward = sum(float(h.get("D_Reward", 0)) for h in episode_history)
    balance_score = 0.5 + 0.5 * (total_d_reward / (total_a_reward + 1e-6))
    balance_score = min(1.0, max(0.0, balance_score))
    
    return {
        "task1_recall": round(task1_score, 3),
        "task2_precision": round(task2_score, 3),
        "task3_hard_mode": round(task3_score, 3),
        "overall_balance": round(balance_score, 3),
        "total_detections": total_detections,
        "total_fns": total_fns,
        "total_fps": total_fps,
        "first_detection_round": first_detection_round if first_detection_round <= num_rounds else "None"
    }

def grader_summary(episode_history: List[Dict], num_rounds: int, difficulty: str = "medium") -> Dict:
    """Return formatted summary for display"""
    scores = run_all_graders(episode_history, num_rounds, difficulty)
    
    return {
        "status": "completed",
        "tasks": {
            "Task 1 (Easy - Detection Recall)": f"{scores['task1_recall']:.3f}",
            "Task 2 (Medium - Pattern Precision)": f"{scores['task2_precision']:.3f}",
            "Task 3 (Hard - Adversarial Resilience & Speed)": f"{scores['task3_hard_mode']:.3f}"
        },
        "overall": {
            "Total True Positives": scores['total_detections'],
            "Total False Negatives": scores['total_fns'],
            "Total False Positives": scores['total_fps'],
            "Speed (First Detect)": f"Round {scores['first_detection_round']}",
            "Balance Score": f"{scores['overall_balance']:.3f}"
        }
    }
    
    return {
        "status": "completed",
        "tasks": {
            "Task 1 (Easy - Accuracy Preservation)": f"{scores['task1_basic_accuracy']:.3f}",
            "Task 2 (Medium - Pattern Recognition)": f"{scores['task2_pattern_recognition']:.3f}",
            "Task 3 (Hard - Robust Defense)": f"{scores['task3_robust_defense']:.3f}"
        },
        "overall": {
            "Final Accuracy": f"{scores['final_accuracy']:.3f}",
            "Balance Score": f"{scores['overall_balance']:.3f}",
            "Total Defender Reward": f"{scores['total_defender_reward']:.2f}",
            "Total Attacker Reward": f"{scores['total_attacker_reward']:.2f}"
        }
    }