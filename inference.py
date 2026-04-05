import asyncio
import os
import json
import textwrap
import sys
import re
import ast
from typing import List, Optional, Dict, Any

# Ensure we can import from server/
sys.path.append(os.path.join(os.getcwd(), "server"))

from openai import OpenAI
from environment import FederatedAdversarialEnv
from models import AttackerAction, DefenderAction
from graders import run_all_graders

# Configuration from Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

TASK_NAME = os.getenv("TASK_NAME", "recall-evaluation")
BENCHMARK = os.getenv("BENCHMARK", "sharperner-rl-v1")
MAX_ROUNDS = 10
DIFFICULTY = os.getenv("DIFFICULTY", "medium").lower()

# Mandatory OpenAI Client
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_llm_action(role: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Call the model and extract JSON action"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=600,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Extract JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        extracted = json_match.group() if json_match else text
        
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(extracted)
            except:
                return {}
            
    except Exception as e:
        print(f"[DEBUG] {role} LLM failure: {e}", flush=True)
        return {}

async def main():
    env = FederatedAdversarialEnv(num_clients=20, num_rounds=MAX_ROUNDS)
    # Target random poisoners for the evaluation
    poisoners = [3, 7, 12]
    env.set_config(poisoners, MAX_ROUNDS, difficulty=DIFFICULTY)
    
    obs = env.reset()
    
    # SMART CONTEXT: Persistent Rolling Memory for both agents
    red_context = "No past actions yet."
    blue_context = "No previous rounds yet."
    
    rewards: List[float] = []
    history: List[Dict] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        for r in range(1, MAX_ROUNDS + 1):
            # Pre-step observation for agents
            light_obs = obs.dict() if hasattr(obs, 'dict') else obs.copy()
            if "client_updates" in light_obs:
                del light_obs["client_updates"]

            # 1. ATTACKER ACTS (LLM powered)
            atk_system = """You are a sophisticated Federated Learning attacker (Red Team). 
Your goal is to reduce the global model accuracy as much as possible using stealthy and effective poisoning techniques.
CRITICAL: You MUST adapt based on previous feedback! If a strategy isn't yielding rewards, or if you get caught, drastically change your `attack_type`, `target_count`, and `strength`!
Available actions: single, coordinated, alie, stealth."""
            
            atk_user = f"""Round: {r} | Acc: {light_obs.get('global_accuracy')} | Telemetry Summary: {light_obs.get('telemetry', {})}
Past Performance Recap: {red_context}
Return ONLY valid JSON: {{"attack_type": "...", "target_count": int, "strength": float, "stealth_level": float, "reasoning": "..."}}"""
            
            atk_data = get_llm_action("Attacker", atk_system, atk_user)
            
            # Attacker Action Mapping
            num_clients = light_obs.get("telemetry", {}).get("total_clients", 20)
            target_count = atk_data.get("target_count", 3)
            import random
            atkTgts = random.sample(range(num_clients), min(target_count, num_clients))
            
            attacker_action = AttackerAction(
                target_clients=atkTgts,
                attack_type=atk_data.get("attack_type", "stealth"),
                strength=float(atk_data.get("strength", 0.8)),
                stealth_level=float(atk_data.get("stealth_level", 0.75)),
                metadata={"reasoning": atk_data.get("reasoning", "")}
            )

            # 2. DEFENDER ACTS (LLM powered)
            def_system = """You are a skilled Federated Learning defender. Detect and stop malicious clients.
CRITICAL: Pay close attention to 'flagged_anomalies_for_defender' in the telemetry!"""
            
            clean_telemetry = {"flagged_anomalies_for_defender": light_obs.get("telemetry", {}).get("flagged_anomalies_for_defender", [])}
            def_user = f"""Round: {r} | Acc: {light_obs.get('global_accuracy')} | Anomaly Data: {clean_telemetry}
Past Performance Recap: {blue_context}
Return ONLY valid JSON: {{"action_type": "quarantine", "target_clients": [ids], "confidence": float, "explanation": "..."}}"""
            
            def_data = get_llm_action("Defender", def_system, def_user)
            
            defender_action = DefenderAction(
                action_type=def_data.get("action_type", "detect"),
                target_clients=def_data.get("target_clients", []),
                confidence=float(def_data.get("confidence", 0.7)),
                explanation=def_data.get("explanation", "")
            )

            # 3. ENVIRONMENT STEP
            observation, a_rew, d_rew, info = env.step(attacker_action, defender_action)
            
            # 4. LOGGING & PROGRESS
            action_summary = f"{defender_action.action_type}:{defender_action.target_clients}"
            log_step(step=r, action=action_summary, reward=d_rew, done=observation.done, error=None)
            
            rewards.append(d_rew)
            steps_taken = r
            
            # Round data for history (graders)
            history.append({
                "Round": r,
                "Attacker Action": f"{attacker_action.attack_type} on {attacker_action.target_clients}",
                "Correct Detections": info.get("correct_detections", 0),
                "False Negatives": info.get("attacker_breach", 0),
                "False Positives": info.get("false_positives", 0),
                "A_Reward": a_rew,
                "D_Reward": d_rew
            })

            # SMART CONTEXT UPDATE (Persistent Rolling Memory)
            # Prepend new round status to old history, keeping total context lean
            atk_status = f"Hits:{info.get('attacker_breach',0)}|Caught:{info.get('correct_detections',0)}|Rew:{a_rew:.1f}"
            red_context = (atk_status + " | " + red_context[:150])
            if len(red_context) > 250: red_context = red_context[:250] + "..."

            def_status = f"Caught:{info.get('correct_detections',0)}|FPs:{info.get('false_positives',0)}|Missed:{info.get('attacker_breach',0)}|Rew:{d_rew:.1f}"
            blue_context = (def_status + " | " + blue_context[:150])
            if len(blue_context) > 250: blue_context = blue_context[:250] + "..."
            
            obs = observation
            if observation.done:
                break

        # 5. FINAL SCORE
        score_results = run_all_graders(history, r)
        final_score = score_results.get("task1_recall", 0.0)
        success = final_score >= 0.1

    except Exception as e:
        print(f"[DEBUG] Simulation critical failure: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
