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

# Multi-provider LLM Router with Fallback
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Primary provider configuration
PRIMARY_PROVIDER = "HuggingFace" if HF_TOKEN else "Groq"

# HuggingFace config (primary if available)
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_CLIENT = OpenAI(base_url=HF_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

# Groq config (fallback or primary)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_CLIENT = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Validate that at least one provider is available
if not HF_CLIENT and not GROQ_CLIENT:
    import sys
    print("[ERROR] No LLM provider configured! Set either HF_TOKEN or GROQ_API_KEY", flush=True)
    sys.exit(1)

# Warn if HF is primary but Groq is not available for fallback
if HF_CLIENT and not GROQ_CLIENT:
    print("[WARNING] HuggingFace set as primary but GROQ_API_KEY not configured. If HF quota is exceeded (402), requests will fail!", flush=True)

# Use primary, fallback to secondary
if PRIMARY_PROVIDER == "HuggingFace" and HF_CLIENT:
    client = HF_CLIENT
    MODEL_NAME = HF_MODEL
    API_BASE_URL = HF_BASE_URL
elif GROQ_CLIENT:
    client = GROQ_CLIENT
    MODEL_NAME = GROQ_MODEL
    API_BASE_URL = GROQ_BASE_URL
else:
    raise ValueError("No valid LLM provider configured. Set HF_TOKEN or GROQ_API_KEY.")

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

def get_llm_response(messages: list, temperature: float = 0.7, max_tokens: int = 600) -> str:
    """Core LLM communication - auto-fallback from HF to Groq on 402/quota errors"""
    global hf_failed, client, MODEL_NAME
    
    # Ensure message content is string
    safe_messages = []
    for msg in messages:
        m = dict(msg)
        if not isinstance(m.get("content"), str):
            m["content"] = str(m.get("content", ""))
        safe_messages.append(m)

    # Try HuggingFace first (if not already failed and available)
    if HF_CLIENT and not hf_failed:
        try:
            completion = HF_CLIENT.chat.completions.create(
                model=HF_MODEL,
                messages=safe_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            
            # Extract JSON if needed
            if "{" in text and "}" in text:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    return match.group()
            return text
            
        except Exception as e:
            error_str = str(e)
            # Detect quota/billing errors (402, 410, 429, out of credits, payment required)
            # 410 = endpoint deprecated/gone; fallback to Groq immediately
            is_quota_error = any(keyword in error_str for keyword in ["402", "410", "429", "Payment", "credits", "depleted", "quota", "limit", "no longer supported"])
            
            if is_quota_error:
                print(f"[WARNING] HuggingFace quota/endpoint error, switching to Groq fallback. Error: {error_str}", flush=True)
                hf_failed = True
                # Continue to Groq fallback below
            else:
                print(f"[DEBUG] HuggingFace error: {error_str}", flush=True)
                if not GROQ_CLIENT:
                    return "Failed to get LLM response."
                # Continue to Groq fallback below

    # Try Groq as fallback or primary
    if GROQ_CLIENT:
        try:
            completion = GROQ_CLIENT.chat.completions.create(
                model=GROQ_MODEL,
                messages=safe_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            
            # Extract JSON if needed
            if "{" in text and "}" in text:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    return match.group()
            return text
            
        except Exception as e:
            print(f"[DEBUG] Groq error: {e}", flush=True)
            return "Failed to get LLM response."
    
    return "Failed to get LLM response."

def get_llm_action(role: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Generates action JSON with automatic HF->Groq fallback"""
    global hf_failed
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Try HuggingFace first (if not already failed)
    if HF_CLIENT and not hf_failed:
        try:
            completion = HF_CLIENT.chat.completions.create(
                model=HF_MODEL,
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
            error_str = str(e)
            # Detect quota/billing errors and 410 endpoint deprecation
            is_quota_error = any(keyword in error_str for keyword in ["402", "410", "429", "Payment", "credits", "depleted", "quota", "limit", "no longer supported"])
            
            if is_quota_error:
                print(f"[WARNING] HuggingFace quota/endpoint error for {role}, switching to Groq. Error: {error_str}", flush=True)
                hf_failed = True
            else:
                print(f"[DEBUG] HuggingFace error for {role}: {e}", flush=True)

    # Try Groq as fallback or primary
    if GROQ_CLIENT:
        try:
            completion = GROQ_CLIENT.chat.completions.create(
                model=GROQ_MODEL,
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
            print(f"[DEBUG] Groq error for {role}: {e}", flush=True)
            return {}
    
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
Your goal is to reduce accuracy while staying stealthy. Adapt based on feedback!"""
            atk_user = f"""Round: {r} | Acc: {obs.global_accuracy:.3f} | Feedback: {red_context}
Return JSON: {{"attack_type": "single|coordinated|alie|stealth", "target_count": int, "strength": float, "stealth_level": float, "reasoning": "..."}}"""
            
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
Return JSON: {{"action_type": "quarantine", "target_clients": [ids], "confidence": float, "explanation": "..."}}"""
            
            def_data = get_llm_action("Defender", def_system, def_user)

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
                    "target_clients": def_data.get("target_clients", []),
                    "confidence": float(def_data.get("confidence", 0.7)),
                    "explanation": def_data.get("explanation", "")
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
                defender_targets=def_data.get('target_clients', []),
                attacker_reward=info.get("attacker_reward", 0.0),
                defender_reward=reward,
                done=obs.done, 
                error=None
            )
            
            rewards_list.append(reward)
            steps_taken = r
            
            # Record for final grading
            history.append({
                "Round": r,
                "Correct Detections": info.get("correct_detections", 0),
                "False Negatives": info.get("attacker_breach", 0),
                "False Positives": info.get("false_positives", 0),
                "A_Reward": info.get("attacker_reward", 0),
                "D_Reward": reward
            })

            # Update contexts for next round (Smart Context)
            atk_status = f"Hits:{info.get('attacker_breach',0)}|Caught:{info.get('correct_detections',0)}|Rew:{info.get('attacker_reward',0):.1f}"
            red_context = (atk_status + " | " + red_context[:150])[:250]
            
            def_status = f"Caught:{info.get('correct_detections',0)}|FPs:{info.get('false_positives',0)}|Missed:{info.get('attacker_breach',0)}|Rew:{reward:.1f}"
            blue_context = (def_status + " | " + blue_context[:150])[:250]

            if obs.done:
                break

        # 6. FINAL SCORING (Deterministic via graders.py logic imported on the client or from server)
        # For submission, the grader should ideally be accessible via /state or /grade.
        # Here we calculate score based on history collected.
        def calculate_score(hist, num_rds):
            tps = sum(h["Correct Detections"] for h in hist)
            fns = sum(h["False Negatives"] for h in hist)
            recall = tps / (tps + fns) if (tps + fns) > 0 else 0
            return min(1.0, recall / 0.8) # Task 1 Recall score

        final_score = calculate_score(history, steps_taken)
        success = final_score >= 0.1

    except Exception as e:
        print(f"[DEBUG] Simulation error: {e}", flush=True)
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards_list)

if __name__ == "__main__":
    asyncio.run(main())
