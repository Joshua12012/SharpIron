
"""
Custom Reward Functions for Red-Blue Adversarial FL
Following your exact specifications
"""

def calculate_attacker_reward(action, success_metrics, round_num):

    reward = 0.0
    reason = []

    # Accuracy drop reward REMOVED as per request

    # NEW: Breach Reward (Increased to 5.0)
    # Every poisoned node that the Defender MISSED is a success for the Red Team.
    attacker_breach = success_metrics.get("attacker_breach", 0)
    if attacker_breach > 0:
        breach_reward = 5.0 * attacker_breach
        reward += breach_reward
        reason.append(f"Uncaught Breaches: +{breach_reward:.1f}")

    # Attack quality bonuses (Increased to 5.0)
    if success_metrics.get("stealth_success", False):
        reward += 5.0
        reason.append("Stealth success +5.0")
    
    if success_metrics.get("coordinated_success", False):
        reward += 5.0
        reason.append("Coordinated success +5.0")
    
    if success_metrics.get("alie_success", False):
        reward += 5.0
        reason.append("ALIE success +5.0")

    if success_metrics.get("imitation_success", False):
        reward += 5.0
        reason.append("Imitation success +5.0")

    if success_metrics.get("single_success", False):
        reward += 5.0
        reason.append("Single client success +5.0")

    # MISSION SUCCESS BONUS: If the model is already falling apart, 
    # keep the pressure on for even more reward.
    current_acc = success_metrics.get("global_accuracy", 1.0)
    if current_acc < 0.15:
        reward += 8.0
        reason.append("Mission Success bonus +8.0")

    # Penalty for being detected (Increased to -5.0)
    correct_detections = success_metrics.get("correct_detections", 0)
    if correct_detections > 0:
        penalty = 2.5 * correct_detections
        reward -= penalty
        reason.append(f"Detection penalty -{penalty:.1f}")

    # Final clamp REMOVED to allow pure learning
    return float(reward), " | ".join(reason) if reason else "No significant impact"


def calculate_defender_reward(action, success_metrics, round_num):

    reward = 0.0
    reason = []

    correct_detections = success_metrics.get("correct_detections", 0)
    false_positives = success_metrics.get("false_positives", 0)

    # Correct attacker detection
    if correct_detections > 0:
        reward += 5.0 * correct_detections
        reason.append(f"Correct detections: +{5.0 * correct_detections:.1f}")

    # Fast detection bonus (earlier = better)
    detection_speed = success_metrics.get("detection_round", round_num)
    fast_bonus = max(0, 10.0 - (detection_speed * 1.5))   # Strong early bonus
    reward += fast_bonus
    if fast_bonus > 5:
        reason.append(f"Fast detection bonus: +{fast_bonus:.1f}")

    # Penalty for accuracy drop REMOVED

    # Penalty for false positives
    if false_positives > 0:
        reward -= 2.5 * false_positives
        reason.append(f"False positives: -{2.5 * false_positives:.1f}")

    # Bonus for correct detection (Reduced to 2.5)
    if correct_detections > 0:
        reward += 2.5
        reason.append("Detection Bonus: +2.5")
    
    # Final clamp REMOVED
    return float(reward), " | ".join(reason) if reason else "Limited performance"