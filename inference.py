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

# Configuration for Evaluation Validator
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

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
    # Tasks to run
    tasks_to_run = [
        {"id": "task1", "name": "Detection Recall (Easy)", "difficulty": "easy"},
        {"id": "task2", "name": "Pattern Precision (Medium)", "difficulty": "medium"},
        {"id": "task3", "name": "Adversarial Resilience (Hard)", "difficulty": "hard"}
    ]

    # Initialize environment
    base_url = os.getenv("PING_URL", "http://localhost:8000")
    env = SharpernerEnv(base_url=base_url)

    try:
        for t_cfg in tasks_to_run:
            task_id = t_cfg["id"]
            task_name = t_cfg["name"]
            difficulty = t_cfg["difficulty"]
            
            # Reset trackers for each task
            red_context = "No past actions yet."
            blue_context = "No previous rounds yet."
            rewards_list: List[float] = []
            history: List[Dict] = []
            steps_taken = 0

            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            # 1. RESET for current task
            reset_result = await env.reset({
                "num_clients": 20,
                "num_rounds": MAX_ROUNDS,
                "poisoners": [3, 7, 12],
                "difficulty": difficulty
            })
            obs = reset_result.observation

            for r in range(1, MAX_ROUNDS + 1):
                if obs.done:
                    break

                # 2. GENERATE ATTACK (LLM)
                atk_system = "You are a sophisticated Federated Learning attacker (Red Team)."
                atk_user = f"Round: {r} | Acc: {obs.global_accuracy:.3f} | Feedback: {red_context}\nReturn action JSON."
                atk_data = get_llm_action("Attacker", atk_system, atk_user)
                
                import random
                num_clients = obs.telemetry.get("total_clients", 20)
                atkTgts = random.sample(range(num_clients), min(atk_data.get("target_count", 3), num_clients))
                
                # 3. GENERATE DEFENSE (LLM)
                def_system = "You are a skilled Federated Learning defender."
                clean_telemetry = {"flagged_anomalies_for_defender": obs.telemetry.get("flagged_anomalies_for_defender", [])}
                def_user = f"Round: {r} | Acc: {obs.global_accuracy:.3f} | Anomalies: {clean_telemetry} | Feedback: {blue_context}\nReturn action JSON."
                def_data = get_llm_action("Defender", def_system, def_user)

                defTgts = def_data.get("target_clients", clean_telemetry.get("flagged_anomalies_for_defender", []))

                # 4. STEP
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
                        "explanation": str(def_data.get("explanation", "Scanning"))
                    }
                )

                result = await env.step(action)
                obs = result.observation
                reward = result.reward
                info = result.info

                # 5. LOG STEP (Baseline Format)
                action_type = def_data.get("action_type", "detect").upper()
                action_str = f"{action_type}{defTgts}".replace(" ", "")
                log_step(step=r, action=action_str, reward=reward, done=obs.done, error=None)
                
                rewards_list.append(reward)
                steps_taken = r
                history.append({
                    "Round": r,
                    "Attacker Action": f"{atk_data.get('attack_type', 'coordinated')} on {atkTgts}",
                    "Defender Action": f"{def_data.get('action_type', 'detect')} on {defTgts}",
                    "Correct Detections": info.get("correct_detections", 0),
                    "False Negatives": info.get("attacker_breach", 0),
                    "False Positives": info.get("false_positives", 0),
                    "A_Reward": info.get("attacker_reward", 0.0),
                    "D_Reward": reward
                })

                if obs.done:
                    break

            # 6. FINAL SCORING for the current task
            try:
                from graders import GRADERS
                # Select the specific grader for this task
                grader_fn = GRADERS.get(task_id)
                if grader_fn:
                    task_score = grader_fn(history, MAX_ROUNDS)
                else:
                    task_score = 0.0
                
                threshold = 0.5 if difficulty == "hard" else 0.7
                success = task_score >= threshold
                log_end(success=success, steps=steps_taken, score=task_score, rewards=rewards_list)
            except Exception as e:
                print(f"[DEBUG] Grading error for {task_id}: {e}", flush=True)
                log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards_list)

    except Exception as e:
        print(f"[DEBUG] Simulation critical error: {e}", flush=True)
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(main())
