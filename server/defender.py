# from models import DefenderAction
# from local_llm import get_llm_response
# import json
# from typing import Dict, Any


# class DefenderAgent:
#     def __init__(self):
#         self.name = "Blue_Defender"
#         self.detection_history = []

#     def act(self, observation: Dict[str, Any]) -> DefenderAction:
#         """Use Groq LLM to analyze telemetry and decide defense action"""
        
#         prompt = f"""You are a skilled Federated Learning defender (Blue Team).
#             Current round: {observation.get('round', 1)}
#             Current global accuracy: {observation.get('global_accuracy', 0.85):.3f}

#             Analyze the following telemetry and decide the best defense action.

#             Telemetry:
#             {observation.get('telemetry', {})}

#             Available actions:
#             - detect: Investigate suspicious clients
#             - quarantine: Remove suspicious clients from aggregation
#             - reweight: Reduce weight of suspicious updates
#             - investigate: Deep dive into patterns
#             - ignore: Do nothing

#             Return ONLY valid JSON with this exact format:
#             {{
#             "action_type": "detect|quarantine|reweight|investigate|ignore",
#             "target_clients": [list of client indices],
#             "confidence": float between 0.5 and 1.0,
#             "explanation": "brief reasoning"
#             }}

#             Be precise and strategic.
#             """

#         response_text = get_llm_response([{"role": "user", "content": prompt}], temperature=0.6, max_tokens=800)

#         try:
#             import re
#             json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#             if json_match:
#                 data = json.loads(json_match.group())
#             else:
#                 data = json.loads(response_text)
#         except:
#             # Fallback
#             data = {
#                 "action_type": "detect",
#                 "target_clients": [0, 1],
#                 "confidence": 0.75,
#                 "explanation": "Fallback due to parsing error"
#             }

#         action = DefenderAction(
#             action_type=data.get("action_type", "detect"),
#             target_clients=data.get("target_clients", []),
#             confidence=float(data.get("confidence", 0.7)),
#             explanation=data.get("explanation", "Analyzing telemetry patterns")
#         )

#         self.detection_history.append(action)
#         return action

#     def update_feedback(self, reward: float, reason: str, target_clients: list):
#         # Stub for compatibility with app.py if needed
#         pass

# defender_agent.py
"""
Blue Defender Agent - Compressed context, fast model only
"""

from models import DefenderAction
from local_llm import get_llm_response
import json
from typing import Dict, Any, List


class DefenderAgent:
    def __init__(self):
        self.system_prompt = """You are a skilled Federated Learning defender. 
Your main goal is to detect and stop malicious clients from sending poisoned updates.
CRITICAL: Pay close attention to the 'flagged_anomalies_for_defender' in the telemetry! These are the statistical outliers this round. You should actively quarantine or detect these specific clients, rather than fixating on old clients from previous rounds."""

        self.context_summary = "No previous rounds yet."

    def reset(self):
        """Reset agent memory for a fresh episode"""
        self.context_summary = "No previous rounds yet."

    def act(self, observation: Dict[str, Any]) -> DefenderAction:
        # FIX: The weaker model is crashing out and dropping to the [0, 1] fallback 
        # because passing two 30-element arrays of floats overwhelms its context.
        # We strip the raw floats and only pass the cleanly digested flagged_anomalies list.
        full_telemetry = observation.get('telemetry', {})
        clean_telemetry = {
            "flagged_anomalies_for_defender": full_telemetry.get("flagged_anomalies_for_defender", [])
        }
        
        current_state = f"""
Round: {observation.get('round')}
Global Accuracy: {observation.get('global_accuracy', 0.85):.3f}
Telemetry: {clean_telemetry}
"""

        user_prompt = f"""Current situation:
{current_state}

Past performance summary:
{self.context_summary}

Decide your defense action! DO NOT copy the example. 
Pick your targets ONLY from the 'flagged_anomalies_for_defender' list.
Return ONLY valid JSON:
{{
"action_type": "quarantine",
"target_clients": [5, 12],
"confidence": 0.9,
"explanation": "reasoning"
}}
"""

        response_text = get_llm_response(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=0.65,
            max_tokens=500
        )

        try:
            import re
            import ast
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            extracted = json_match.group() if json_match else response_text
            try:
                data = json.loads(extracted)
            except json.JSONDecodeError:
                # Fallback to ast if the LLM output single quotes (like a python dict)
                data = ast.literal_eval(extracted)
        except Exception as e:
            print(f"\n[DEFENDER JSON PARSE ERROR] {e}\nRaw LLM Output:\n{response_text}\n")
            data = {"action_type": "detect", "target_clients": [0,1], "confidence": 0.7, "explanation": "Fallback"}

        action = DefenderAction(
            action_type=data.get("action_type", "detect"),
            target_clients=data.get("target_clients", []),
            confidence=float(data.get("confidence", 0.7)),
            explanation=data.get("explanation", "")
        )

        return action

    def update_feedback(self, reward: float, reason: str, detections: int = 0, fps: int = 0, missed: int = 0):
        """Compress feedback into a structured success/failure string to save tokens"""
        # We ignore the verbose 'reason' string for context to stay lean
        status = f"Caught:{detections}|FPs:{fps}|missed:{missed}|Rew:{reward:.1f}"
        
        self.context_summary = status + " | " + self.context_summary[:150]
        
        if len(self.context_summary) > 250:
            self.context_summary = self.context_summary[:250] + "..."