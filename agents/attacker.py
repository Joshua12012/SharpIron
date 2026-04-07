# from models import AttackerAction
# from local_llm import get_llm_response
# import json
# from typing import Dict, Any


# class AttackerAgent:
#     def __init__(self):
#         self.name = "Red_Attacker"
#         self.attack_history = []

#     def act(self, observation: Dict[str, Any], difficulty: str = "medium") -> AttackerAction:
#         """Use Groq LLM to decide smart attack strategy"""
        
#         prompt = f"""You are a sophisticated Federated Learning attacker (Red Team).
#             Current round: {observation.get('round', 1)}
#             Current global accuracy: {observation.get('global_accuracy', 0.85):.3f}
#             Difficulty: {difficulty}

#             Your goal is to reduce the global model accuracy as much as possible while staying stealthy.

#             Available attack types:
#             - single: Attack one client heavily
#             - coordinated: Attack multiple clients together
#             - alie: "A Little Is Enough" - subtle coordinated attack
#             - stealth: Very hidden, slow poisoning

#             Decide the best attack for this round.
#             Return ONLY valid JSON with this exact format:
#             {{
#             "attack_type": "single|coordinated|alie|stealth",
#             "target_count": integer between 1 and 8,
#             "strength": float between 0.6 and 1.0,
#             "stealth_level": float between 0.3 and 0.95,
#             "reasoning": "brief explanation"
#             }}

#             Current telemetry summary: {observation.get('telemetry', {})}
#             """

#         response_text = get_llm_response([{"role": "user", "content": prompt}], temperature=0.8, max_tokens=800)

#         try:
#             # Extract JSON from LLM response
#             import re
#             json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#             if json_match:
#                 data = json.loads(json_match.group())
#             else:
#                 data = json.loads(response_text)
#         except:
#             # Fallback if JSON parsing fails
#             data = {
#                 "attack_type": "coordinated",
#                 "target_count": 3,
#                 "strength": 0.85,
#                 "stealth_level": 0.7,
#                 "reasoning": "Fallback due to parsing error"
#             }

#         # Select random clients
#         num_clients = len(observation.get("client_updates", []))
#         target_clients = list(range(num_clients))
#         import random
#         target_clients = random.sample(target_clients, min(data.get("target_count", 3), num_clients))

#         action = AttackerAction(
#             target_clients=target_clients,
#             attack_type=data.get("attack_type", "coordinated"),
#             strength=float(data.get("strength", 0.85)),
#             stealth_level=float(data.get("stealth_level", 0.7)),
#             metadata={
#                 "round": observation.get("round", 1),
#                 "difficulty": difficulty,
#                 "llm_reasoning": data.get("reasoning", "")
#             }
#         )

#         self.attack_history.append(action)
#         return action
    
#     def update_feedback(self, reward: float, reason: str):
#         # Stub for compatibility with app.py if needed
#         pass
# attacker_agent.py
"""
Red Attacker Agent - Compressed context, fast model only
"""

# attacker_agent.py
"""
Red Attacker Agent with proper persistent context for learning from mistakes
"""

from models import AttackerAction
from inference import get_llm_response
import json
from typing import Dict, Any, List


class AttackerAgent:
    def __init__(self):
        self.name = "Red_Attacker"
        
        self.system_prompt = """You are a sophisticated Federated Learning attacker (Red Team). 
            Your goal is to reduce the global model accuracy as much as possible using stealthy and effective poisoning techniques.
            CRITICAL: You MUST adapt based on previous feedback! If a strategy isn't yielding sharply increasing rewards, or if you get caught, drastically change your `attack_type`, `target_count`, and `strength`! Do not endlessly repeat the same strategy over and over.

            You can use imitation (acting like benign clients) to regain trust when heavily detected.

            ADVISORY: Being caught now carries a heavy -5.0 penalty per node. 
            - 'stealth' and 'single' node attacks are much safer for long-term survival.
            - 'coordinated' and 'alie' carry high reward but high risk.

            Available attack types:
            - single: Attack one client heavily
            - coordinated: Attack multiple clients together
            - alie: "A Little Is Enough" - subtle coordinated attack
            - stealth: Very hidden, slow poisoning
            """
        
        self.context_summary = "No past actions yet."

    def reset(self):
        """Reset agent memory for a fresh episode"""
        self.context_summary = "No past actions yet."

    def act(self, observation: Dict[str, Any], difficulty: str = "medium") -> AttackerAction:
        """Decide attack strategy using LLM with compressed context"""
                    
        user_prompt = f"""Current situation:
            Round: {observation.get('round')}
            Global Accuracy: {observation.get('global_accuracy', 0.85)}
            Difficulty: {difficulty}
            Telemetry summary: {observation.get('telemetry', {})}

            Past performance summary:
            {self.context_summary}

            Decide your attack action. Return ONLY valid JSON with this exact format:
            {{
            "attack_type": "single|coordinated|alie|stealth",
            "target_count": integer between 1 and 8,
            "strength": float between 0.6 and 1.0,
            "stealth_level": float between 0.3 and 0.95,
            "reasoning": "brief explanation"
            }}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response_text = get_llm_response(messages, temperature=0.75, max_tokens=700)

        # Parse JSON response
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response_text)
        except:
            # Fallback strategy
            data = {
                "attack_type": "stealth",
                "target_count": 3,
                "strength": 0.80,
                "stealth_level": 0.75,
                "reasoning": "Fallback due to parsing error"
            }

        # Select random clients to attack
        num_clients = observation.get("telemetry", {}).get("total_clients", 20)
        import random
        target_clients = random.sample(range(num_clients), min(data.get("target_count", 3), num_clients))

        action = AttackerAction(
            target_clients=target_clients,
            attack_type=data.get("attack_type", "stealth"),
            strength=float(data.get("strength", 0.80)),
            stealth_level=float(data.get("stealth_level", 0.75)),
            metadata={"reasoning": data.get("reasoning", "")}
        )

        return action

    def update_feedback(self, reward: float, reason: str, detections: int = 0, breaches: int = 0):
        """Compress feedback into a structured success/failure string to save tokens"""
        # We ignore the verbose 'reason' string for context to stay lean
        status = f"Hits:{breaches}|Caught:{detections}|Rew:{reward:.1f}"
        
        self.context_summary = status + " | " + self.context_summary[:150]
        
        if len(self.context_summary) > 250:
            self.context_summary = self.context_summary[:250] + "..."