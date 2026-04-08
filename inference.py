import asyncio
import os
import json
import textwrap
import sys
import re
import ast
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from sharperner_env import SharpernerEnv, SharpernerAction

load_dotenv()

# Configuration for Evaluation Validator - Strictly following instructions
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY") 
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Fallbacks for local development if environment variables are not set
if not API_BASE_URL:
    API_BASE_URL = os.getenv("HF_ENDPOINT", "https://router.huggingface.co/v1")
if not API_KEY:
    API_KEY = os.getenv("HF_TOKEN")

# Initialize OpenAI client as requested
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

# Validate provider
USE_DUMMY_LLM = False
if not client:
    print("[DEBUG] No LLM provider configured (API_KEY missing). Falling back to dummy behavior.", flush=True)
    USE_DUMMY_LLM = True

TASK_NAME = os.getenv("TASK_NAME", "recall-evaluation")
BENCHMARK = os.getenv("BENCHMARK", "sharperner-rl-v1")
MAX_ROUNDS = 10
DIFFICULTY = os.getenv("DIFFICULTY", "medium").lower()

# Track provider failures for fallback
hf_failed = False

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, attacker_targets: List[int], defender_targets: List[int], attacker_reward: float, defender_reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} | Atk_Targets={attacker_targets} | Def_Targets={defender_targets} | Rew_A={attacker_reward:.2f} | Rew_D={defender_reward:.2f} | done={done_val} | error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def _dummy_llm_response(messages: list) -> str:
    """Fallback stub when no LLM provider is configured."""
    # Provide a very simple heuristic response that lets the simulation continue.
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    if "attack_type" in last_user.get("content", ""):
        return json.dumps({
            "attack_type": "coordinated",
            "target_count": 2,
            "strength": 0.8,
            "stealth_level": 0.7,
            "reasoning": "Fallback dummy attacker"
        })
    if "action_type" in last_user.get("content", ""):
        return json.dumps({
            "action_type": "detect",
            "target_clients": [0, 1],
            "confidence": 0.75,
            "explanation": "Fallback dummy defender"
        })
    return "{}"


def get_llm_response(messages: list, temperature: float = 0.7, max_tokens: int = 600) -> str:
    """Core LLM communication directly via the active client"""
    if not client:
        return ""
        
    safe_messages = []
    for msg in messages:
        m = dict(msg)
        if not isinstance(m.get("content"), str):
            m["content"] = str(m.get("content", ""))
        safe_messages.append(m)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=safe_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        if "{" in text and "}" in text:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return match.group()
        return text
    except Exception as e:
        print(f"[DEBUG] API error: {e}", flush=True)
        return "Failed to get LLM response."

def get_llm_action(role: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Generates action JSON using the active client"""
    if not client:
        return {}
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=600,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
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
        # To avoid spamming terminal if daily limit is reached
        if 'rate_limit_exceeded' not in str(e):
            print(f"[DEBUG] {role} API error: {e}", flush=True)
        return {}

async def main():
    # Initialize environment via standard /reset API
    # Use the PING_URL from environment or default to localhost
    base_url = os.getenv("PING_URL", "http://localhost:8000")
    env = SharpernerEnv(base_url=base_url)
    
    # Trackers for agents (Smart Context logic)
    red_context = "No past actions yet."
    blue_context = "No previous rounds yet."
    
    rewards_list: List[float] = []
    history: List[Dict] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # 1. RESET (Calls POST /reset on backend)
        reset_result = await env.reset({
            "num_clients": 20,
            "num_rounds": MAX_ROUNDS,
            "poisoners": [3, 7, 12],
            "difficulty": DIFFICULTY
        })
        obs = reset_result.observation

        for r in range(1, MAX_ROUNDS + 1):
            if obs.done:
                break

            # Prepare telemetry for LLM
            light_obs = {
                "round": obs.round,
                "global_accuracy": obs.global_accuracy,
                "telemetry": obs.telemetry
            }

            # 2. GENERATE ATTACK (LLM)
            atk_system = """You are a sophisticated Federated Learning attacker (Red Team). 
Your goal is to reduce accuracy while using your different strategies properly so your reward goes on increasing. Adapt based on feedback!"""
            atk_user = f"""Round: {r} | Acc: {obs.global_accuracy:.3f} | Feedback: {red_context}
Return ONLY valid JSON: {{"attack_type": "stealth", "target_count": 3, "strength": 0.8, "stealth_level": 0.9, "reasoning": "testing stealth"}}"""
            
            atk_data = get_llm_action("Attacker", atk_system, atk_user)
            
            # Map choice to random clients (simulation side)
            import random
            num_clients = obs.telemetry.get("total_clients", 20)
            atkTgts = random.sample(range(num_clients), min(atk_data.get("target_count", 3), num_clients))
            
            # 3. GENERATE DEFENSE (LLM)
            def_system = """You are a skilled Federated Learning defender. Detect and stop malicious clients.
Pay close attention to 'flagged_anomalies_for_defender'!"""
            
            clean_telemetry = {"flagged_anomalies_for_defender": obs.telemetry.get("flagged_anomalies_for_defender", [])}
            def_user = f"""Round: {r} | Acc: {obs.global_accuracy:.3f} | Anomalies: {clean_telemetry} | Feedback: {blue_context}
Return ONLY valid JSON: {{"action_type": "quarantine", "target_clients": [1, 5, 12], "confidence": 0.85, "explanation": "blocking anomalies"}}"""
            
            def_data = get_llm_action("Defender", def_system, def_user)

            defTgts = def_data.get("target_clients", clean_telemetry.get("flagged_anomalies_for_defender", []))

            # 4. STEP (Calls POST /step on backend)
            # We bundle both actions as expected by the new standardized /step API
            action = SharpernerAction(
                attacker_action={
                    "target_clients": atkTgts,
                    "attack_type": atk_data.get("attack_type", "coordinated"),
                    "strength": float(atk_data.get("strength", 0.8)),
                    "stealth_level": float(atk_data.get("stealth_level", 0.7))
                },
                defender_action={
                    "action_type": def_data.get("action_type", "detect"),
                    "target_clients": defTgts,
                    "confidence": float(def_data.get("confidence", 0.7)),
                    "explanation": str(def_data.get("explanation", "Fallback to anomalies"))
                }
            )

            result = await env.step(action)
            obs = result.observation
            reward = result.reward
            info = result.info

            # 5. LOGGING (Detailed Format)
            log_step(
                step=r, 
                attacker_targets=atkTgts,
                defender_targets=defTgts,
                attacker_reward=info.get("attacker_reward", 0.0),
                defender_reward=reward,
                done=obs.done, 
                error=None
            )
            
            rewards_list.append(reward)
            steps_taken = r
            
            # Record for final grading mapped exactly as graders.py expects
            history.append({
                "Round": r,
                "Attacker Action": f"{atk_data.get('attack_type', 'coordinated')} on clients {atkTgts}",
                "Defender Action": f"{def_data.get('action_type', 'detect')} on clients {defTgts}",
                "Correct Detections": info.get("correct_detections", 0),
                "False Negatives": info.get("attacker_breach", 0),
                "False Positives": info.get("false_positives", 0),
                "A_Reward": info.get("attacker_reward", 0.0),
                "D_Reward": reward
            })

            # Update contexts for next round (Smart Context)
            atk_status = f"Hits:{info.get('attacker_breach',0)}|Caught:{info.get('correct_detections',0)}|Rew:{info.get('attacker_reward',0):.1f}"
            red_context = (atk_status + " | " + red_context[:150])[:250]
            
            def_status = f"Caught:{info.get('correct_detections',0)}|FPs:{info.get('false_positives',0)}|Missed:{info.get('attacker_breach',0)}|Rew:{reward:.1f}"
            blue_context = (def_status + " | " + blue_context[:150])[:250]

            if obs.done:
                break

        # 6. FINAL SCORING
        try:
            from graders import grader_summary
            grader_res = grader_summary(history, MAX_ROUNDS, DIFFICULTY)
            final_score = float(grader_res["tasks"].get("Task 1 (Easy - Detection Recall)", 0.0))
            success = final_score >= 0.8
            
            # Log grader results for validator discovery
            for task_name, score in grader_res["tasks"].items():
                print(f"[GRADER] task={task_name} score={score:.3f}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Error running graders: {e}", flush=True)
            # Fallback to manual calculation
            def calculate_score(hist, num_rds):
                tps = sum(h["Correct Detections"] for h in hist)
                fns = sum(h["False Negatives"] for h in hist)
                if (tps + fns) == 0: return 0.0
                recall = tps / (tps + fns)
                return min(1.0, recall / 0.8)

            final_score = calculate_score(history, steps_taken)
            success = final_score >= 0.1

    except Exception as e:
        print(f"[DEBUG] Simulation error: {e}", flush=True)
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards_list)

if __name__ == "__main__":
    asyncio.run(main())
