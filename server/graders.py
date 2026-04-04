# graders.py
"""
Graders for Red-Blue Adversarial Federated Learning Environment
3 tasks with increasing difficulty
"""

from typing import Dict, Any, List

def run_all_graders(episode_history: List[Dict], final_accuracy: float, num_rounds: int) -> Dict:
    """
    Compute 3 graded tasks for the episode.
    Returns scores between 0.0 and 1.0 for each task.
    """
    
    total_attacker_reward = sum(float(h.get("A_Reward", 0)) for h in episode_history)
    total_defender_reward = sum(float(h.get("D_Reward", 0)) for h in episode_history)
    
    avg_accuracy = sum(float(h.get("Accuracy", 0)) for h in episode_history) / len(episode_history)
    
    # Count meaningful defender actions
    defender_actions = [h.get("Defender", "") for h in episode_history]
    detection_count = sum(1 for a in defender_actions if a in ["detect", "quarantine", "investigate"])
    
    import re
    # Task 1: Basic Accuracy Preservation (Easy)
    # Rescaled to 0.40 baseline because the attacker's dynamic damage multiplier was increased to 0.25
    # 0.40 = 0% score, 0.85+ = 100% score
    task1_score = min(1.0, max(0.0, (final_accuracy - 0.40) / 0.45))
    
    # Task 2: True Correct Detections (Medium)
    # We now strictly check if the Defender's target_clients overlap with the Attacker's targets!
    true_detections = 0
    for h in episode_history:
        atk_match = re.search(r'on clients \[([0-9, ]+)\]', h.get("Attacker Action", ""))
        def_match = re.search(r'on clients \[([0-9, ]+)\]', h.get("Defender Action", ""))
        if atk_match and def_match:
            atk_clients = set([int(x) for x in atk_match.group(1).split(",") if x.strip().isdigit()])
            def_clients = set([int(x) for x in def_match.group(1).split(",") if x.strip().isdigit()])
            if len(atk_clients & def_clients) > 0:
                true_detections += 1

    coordinated_defense = sum(1 for h in episode_history if "coordinated" in h.get("Defender Action", "").lower())
    task2_score = min(1.0, (true_detections / max(4, num_rounds * 0.5)) * 0.7 + (coordinated_defense / max(1, num_rounds)) * 0.3)
    
    # Task 3: Fast & Robust Defense (Hard)
    # Reward early detection + low false positives + resilience
    early_detection_bonus = sum(1 for h in episode_history if float(h.get("Accuracy", 1.0)) > 0.80) / num_rounds
    task3_score = min(1.0, early_detection_bonus * 0.6 + (final_accuracy / 0.95) * 0.4)
    
    # Overall balance check
    balance_score = 0.5 + 0.5 * (total_defender_reward / (total_attacker_reward + 1e-6))
    balance_score = min(1.0, max(0.0, balance_score))
    
    return {
        "task1_basic_accuracy": round(task1_score, 3),
        "task2_pattern_recognition": round(task2_score, 3),
        "task3_robust_defense": round(task3_score, 3),
        "overall_balance": round(balance_score, 3),
        "final_accuracy": round(final_accuracy, 3),
        "total_defender_reward": round(total_defender_reward, 2),
        "total_attacker_reward": round(total_attacker_reward, 2)
    }


def grader_summary(episode_history: List[Dict], final_accuracy: float, num_rounds: int) -> Dict:
    """Return formatted summary for display"""
    scores = run_all_graders(episode_history, final_accuracy, num_rounds)
    
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