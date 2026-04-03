
"""
Red Attacker Agent with compact context management
"""

from models import AttackerAction
from local_llm import get_llm_response_with_history
import json

class AttackerAgent:
    def __init__(self):
        self.name = "Red_Attacker"
        self.summary = "I am a sophisticated Federated Learning attacker. My goal is to reduce global model accuracy using stealthy poisoning techniques."
        self.recent_feedback = []   # Keep only last 3 rounds

    def act(self, observation, difficulty="medium"):
        # Build compact user message
        current_state = f"""
        Round: {observation.get('round', 1)}
        Global Accuracy: {observation.get('global_accuracy', 0.85):.3f}
        Difficulty: {difficulty}
        Telemetry summary: {observation.get('telemetry', {})}
        """

                # Add recent feedback (limited)
        feedback_text = "\n".join(self.recent_feedback[-3:]) if self.recent_feedback else "No previous feedback."

        user_prompt = f"""Current situation:
        {current_state}

        Recent performance feedback:
        {feedback_text}

        Decide your next attack. Return ONLY valid JSON:
        {{
        "attack_type": "single|coordinated|alie|stealth|imitation"
        "target_count": integer (1-8),
        "strength": float (0.6-1.0),
        "stealth_level": float (0.3-0.95),
        "reasoning": "short explanation"
        }}
        """

        response_text = get_llm_response_with_history(
            system_prompt=self.summary,
            user_messages=[{"role": "user", "content": user_prompt}]
        )

        try:
            data = json.loads(response_text)
        except:
            data = {"attack_type": "coordinated", "target_count": 3, "strength": 0.85, "stealth_level": 0.7, "reasoning": "Fallback"}

        # Select targets
        num_clients = len(observation.get("client_updates", []))
        import random
        target_clients = random.sample(range(num_clients), min(data.get("target_count", 3), num_clients))

        action = AttackerAction(
            target_clients=target_clients,
            attack_type=data.get("attack_type", "coordinated"),
            strength=float(data.get("strength", 0.85)),
            stealth_level=float(data.get("stealth_level", 0.7)),
            metadata={"reasoning": data.get("reasoning", "")}
        )

        return action

    def update_feedback(self, reward: float, reason: str):
        """Update agent with reward and mistake feedback - compact"""
        feedback = f"Round feedback: Reward = {reward:.2f}. {reason}"
        self.recent_feedback.append(feedback)
        if len(self.recent_feedback) > 3:
            self.recent_feedback.pop(0)   # Keep only last 3