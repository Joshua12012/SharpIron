
# rewards.py
"""
Balanced and realistic reward functions for Red-Blue FL adversarial setup.
Defender only gets high reward when it correctly detects real attackers.
"""

def calculate_attacker_reward(action, success_metrics, round_num):
    reward = 0.0
    reason = []

    accuracy_drop = success_metrics.get("accuracy_drop", 0.0)

    # Main reward: causing actual damage to global model
    reward += 28.0 * accuracy_drop
    if accuracy_drop > 0.015:
        reason.append(f"Caused real damage ({accuracy_drop:.3f})")

    # Bonus for attack quality
    if success_metrics.get("stealth_success", False):
        reward += 9.0
        reason.append("Stealth attack succeeded")
    if success_metrics.get("coordinated_success", False):
        reward += 11.0
        reason.append("Coordinated attack succeeded")
    if success_metrics.get("alie_success", False):
        reward += 10.0
        reason.append("ALIE attack succeeded")

    # Imitation / trust recovery bonus
    if success_metrics.get("imitation_success", False):
        reward += 7.0
        reason.append("Successfully imitated benign clients")

    # Penalty if heavily detected
    if success_metrics.get("detected_count", 0) >= 3:
        reward -= 14.0
        reason.append("Heavily detected by defender")

    reward = max(-18.0, min(40.0, reward))
    return float(reward), " | ".join(reason) if reason else "Minor impact"


def calculate_defender_reward(action, success_metrics, round_num):
    reward = 0.0
    reason = []

    accuracy_drop = success_metrics.get("accuracy_drop", 0.0)
    correct_detections = success_metrics.get("correct_detections", 0)   # NEW: only real poisoned clients flagged
    false_positives = success_metrics.get("false_positives", 0)

    # Strong reward ONLY if defender correctly flagged real attackers
    if correct_detections > 0:
        reward += 15.0 * correct_detections
        reason.append(f"Correctly detected {correct_detections} real attackers")

    # Fast detection bonus (earlier is better)
    detection_speed = success_metrics.get("detection_round", round_num)
    fast_bonus = max(0, 12.0 - detection_speed * 0.9)
    reward += fast_bonus
    if fast_bonus > 3:
        reason.append(f"Fast detection bonus (+{fast_bonus:.1f})")

    # Penalty for allowing damage
    if accuracy_drop > 0.02:
        reward -= 18.0 * (accuracy_drop / 0.02)
        reason.append(f"Failed to prevent damage ({accuracy_drop:.3f})")

    # Penalty for false positives (quarantining innocent clients)
    if false_positives > 0:
        reward -= 8.0 * false_positives
        reason.append(f"False positives: {false_positives} innocent clients flagged")

    # Bonus for maintaining accuracy when no attack happened
    if accuracy_drop < 0.005 and correct_detections == 0:
        reward += 6.0
        reason.append("Successfully maintained accuracy")

    reward = max(-25.0, min(45.0, reward))
    return float(reward), " | ".join(reason) if reason else "Limited effectiveness"