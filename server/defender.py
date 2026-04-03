# defender_agent.py
"""
Blue Defender Agent with compact context management
"""

from models import DefenderAction
from local_llm import get_llm_response_with_history
import json

class DefenderAgent:
    def __init__(self):
        self.name = "Blue_Defender"
        self.summary = "I am a skilled Federated Learning defender. My goal is to detect poisoning attacks early, protect global accuracy, and minimize false positives."
        self.recent_feedback = []

    def act(self, observation):
        current_state = f"""
        Round: {observation.get('round', 1)}
        Global Accuracy: {observation.get('global_accuracy', 0.85):.3f}
        Telemetry: {observation.get('telemetry', {})}
        """

        feedback_text = "\n".join(self.recent_feedback[-3:]) if self.recent_feedback else "No previous feedback."

        user_prompt = f"""Current situation:
        {current_state}

        Recent performance feedback:
        {feedback_text}

        Decide your defense action. Return ONLY valid JSON:
        {{
        "action_type": "detect|quarantine|reweight|investigate|ignore",
        "target_clients": [list of integers],
        "confidence": float (0.5-1.0),
        "explanation": "short reasoning"
        }}
        """

        response_text = get_llm_response_with_history(
            system_prompt=self.summary,
            user_messages=[{"role": "user", "content": user_prompt}]
        )

        try:
            data = json.loads(response_text)
        except:
            data = {"action_type": "detect", "target_clients": [0,1], "confidence": 0.7, "explanation": "Fallback"}

        action = DefenderAction(
            action_type=data.get("action_type", "detect"),
            target_clients=data.get("target_clients", []),
            confidence=float(data.get("confidence", 0.7)),
            explanation=data.get("explanation", "")
        )

        return action

    def update_feedback(self, reward: float, reason: str):
        """Compact feedback update"""
        feedback = f"Reward = {reward:.2f}. {reason}"
        self.recent_feedback.append(feedback)
        if len(self.recent_feedback) > 3:
            self.recent_feedback.pop(0)