
"""
Custom Reward Functions for Red-Blue Adversarial FL
Following your exact specifications
"""

def calculate_attacker_reward(action, success_metrics, round_num):

    reward = 0.0
    reason = []

    accuracy_drop = success_metrics.get("accuracy_drop", 0.0)

    # Accuracy drop reward
    reward += 40.0 * accuracy_drop
    if accuracy_drop > 0.01:
        reason.append(f"Accuracy drop: +{40.0 * accuracy_drop:.2f}")

    # NEW: Breach Reward (Rewriting the rules!)
    # Every poisoned node that the Defender MISSED is a success for the Red Team.
    attacker_breach = success_metrics.get("attacker_breach", 0)
    if attacker_breach > 0:
        breach_reward = 3.0 * attacker_breach
        reward += breach_reward
        reason.append(f"Uncaught Breaches: +{breach_reward:.1f}")

    # Attack quality bonuses
    if success_metrics.get("stealth_success", False):
        reward += 2.5
        reason.append("Stealth success +2.5")

    if success_metrics.get("coordinated_success", False):
        reward += 2.5
        reason.append("Coordinated success +2.5")

    if success_metrics.get("alie_success", False):
        reward += 2.5
        reason.append("ALIE success +2.5")

    if success_metrics.get("imitation_success", False):
        reward += 5.0
        reason.append("Imitation success +5.0")

    # MISSION SUCCESS BONUS: If the model is already falling apart, 
    # keep the pressure on for even more reward.
    current_acc = success_metrics.get("global_accuracy", 1.0)
    if current_acc < 0.15:
        reward += 8.0
        reason.append("Mission Success bonus +8.0")

    # Penalty for being detected by the defender (Balanced to not wipe out all profit)
    correct_detections = success_metrics.get("correct_detections", 0)
    if correct_detections > 0:
        penalty = 3.0 * correct_detections
        reward -= penalty
        reason.append(f"Detection penalty -{penalty:.1f}")

    reward = max(-15.0, min(50.0, reward))
    return float(reward), " | ".join(reason) if reason else "No significant impact"


def calculate_defender_reward(action, success_metrics, round_num):

    reward = 0.0
    reason = []

    accuracy_drop = success_metrics.get("accuracy_drop", 0.0)
    correct_detections = success_metrics.get("correct_detections", 0)
    false_positives = success_metrics.get("false_positives", 0)

    # Correct attacker detection
    if correct_detections > 0:
        reward += 10.0 * correct_detections
        reason.append(f"Correct detections: +{10.0 * correct_detections:.1f}")

    # Fast detection bonus (earlier = better)
    detection_speed = success_metrics.get("detection_round", round_num)
    fast_bonus = max(0, 20.0 - (detection_speed * 1.5))   # Strong early bonus
    reward += fast_bonus
    if fast_bonus > 5:
        reason.append(f"Fast detection bonus: +{fast_bonus:.1f}")

    # Penalty for accuracy drop
    if accuracy_drop > 0:
        reward -= 5.0 * accuracy_drop
        reason.append(f"Accuracy drop penalty: -{5.0 * accuracy_drop:.2f}")

    # Penalty for false positives
    if false_positives > 0:
        reward -= 2.5 * false_positives
        reason.append(f"False positives: -{2.5 * false_positives:.1f}")

    # Bonus for correct detection + accuracy improvement
    if correct_detections > 0 and accuracy_drop < 0.01:
        reward += 10.0
        reason.append("Correct detection + accuracy maintained: +10.0")

    reward = max(-25.0, min(45.0, reward))
    return float(reward), " | ".join(reason) if reason else "Limited performance"