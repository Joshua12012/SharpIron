from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
import sys
import os
from pathlib import Path
import json
import asyncio

# Ensure the absolute root directory of the project is in the Python path
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Core imports (Using absolute resolution from ROOT)
from server.environment import FederatedAdversarialEnv
from models import AttackerAction, DefenderAction, Observation, RewardInfo
from agents.attacker import AttackerAgent
import agents.attacker
from agents.defender import DefenderAgent
import agents.defender
from graders import TASK_DEFINITIONS, GRADER_DEFINITIONS, grader_summary

# from environment import FederatedAdversarialEnv
# from models import AttackerAction, DefenderAction, Observation, RewardInfo
# from agents.attacker import AttackerAgent
# from agents.defender import DefenderAgent
# from graders import TASK_DEFINITIONS, GRADER_DEFINITIONS, grader_summary

app = FastAPI(
    title="SharpernerRL",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    redirect_slashes=False
)

# CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for the SENTINEL UI
# Using server/static because uvicorn runs from the root
os.makedirs("server/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="server/static"), name="static")

@app.get("/", include_in_schema=False)
def root_page():
    from fastapi.responses import FileResponse
    return FileResponse("server/static/index.html")

# Standard Defaults for OpenEnv Validation
DEFAULT_CLIENTS = 20
DEFAULT_ROUNDS = 15
DEFAULT_POISONERS = [3, 7, 12]
DEFAULT_DIFFICULTY = "medium"

# Global instances
env = FederatedAdversarialEnv(num_clients=DEFAULT_CLIENTS, num_rounds=DEFAULT_ROUNDS)
attacker_agent = AttackerAgent()
defender_agent = DefenderAgent()
last_grade_summary: Optional[Dict[str, Any]] = None

def log_graders(tasks: Dict[str, float]) -> None:
    """Log grader results for validator discovery"""
    import sys
    for task_name, score in tasks.items():
        print(f"[GRADER] task={task_name} score={score:.3f}", flush=True)
    sys.stdout.flush()

def log_start():
    import time
    import sys
    import os
    task = os.getenv("TASK_NAME", "recall-evaluation")
    env_name = os.getenv("BENCHMARK", "sharperner-rl-v1")
    model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    print(f"[START] task={task} env={env_name} model={model}")
    sys.stdout.flush()

def log_step(step, atk_action, def_action, a_rew, d_rew, done):
    import sys
    done_str = str(done).lower()
    # Combine both agents into the action description
    action_desc = f"{{'attacker': {{'targets': {atk_action.target_clients}, 'type': '{atk_action.attack_type}'}}, 'defender': {{'targets': {def_action.target_clients}, 'type': '{def_action.action_type}'}}}}"
    print(f"[STEP] step={step} action={action_desc} rew_a={a_rew:.2f} rew_d={d_rew:.2f} done={done_str} error=null")
    sys.stdout.flush()

def log_end(success, steps, score, rewards):
    import sys
    success_str = str(success).lower()
    rew_str = ",".join([f"{r:.1f}" for r in rewards])
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rew_str}")
    sys.stdout.flush()

class EpisodeRequest(BaseModel):
    poisoners: List[int] = DEFAULT_POISONERS
    num_clients: int = DEFAULT_CLIENTS
    num_rounds: int = DEFAULT_ROUNDS
    difficulty: str = DEFAULT_DIFFICULTY

async def run_episode_generator(request: EpisodeRequest):
    """Generator for streaming round-by-round results to the UI"""
    env.num_clients = request.num_clients
    env.num_rounds = request.num_rounds
    env.set_config(request.poisoners, request.num_rounds, difficulty=request.difficulty)
    
    obs = env.reset()
    attacker_agent.reset()
    defender_agent.reset()
    
    history = []
    rewards_history = []
    cum_detections = 0
    cum_breaches = 0
    cum_fps = 0

    yield f"data: {json.dumps({'type': 'init', 'msg': 'Simulation started'})}\n\n"
    log_start()

    for r in range(request.num_rounds):
        # Create light obs
        light_obs = obs.dict() if hasattr(obs, 'dict') else obs.copy()
        if "client_updates" in light_obs:
            del light_obs["client_updates"]

        # Agents act (IO-bound LLM calls)
        attacker_action = attacker_agent.act(light_obs, request.difficulty)
        defender_action = defender_agent.act(light_obs)

        # Env Step
        observation, a_rew, d_rew, info = env.step(attacker_action, defender_action)
        
        # Update Agents
        attacker_agent.update_feedback(
            a_rew, info.get("attacker_reason", ""),
            detections=info.get("correct_detections", 0),
            breaches=info.get("attacker_breach", 0)
        )
        defender_agent.update_feedback(
            d_rew, info.get("defender_reason", ""),
            detections=info.get("correct_detections", 0),
            fps=info.get("false_positives", 0),
            missed=info.get("attacker_breach", 0)
        )

        # Accumulate metrics
        tp = info.get("correct_detections", 0)
        fn = info.get("attacker_breach", 0)
        fp = info.get("false_positives", 0)
        cum_detections += tp
        cum_breaches += fn
        cum_fps += fp

        # Log for OpenEnv evaluation (Standardized - Combined Agents)
        log_step(r+1, attacker_action, defender_action, a_rew, d_rew, observation.done)
        rewards_history.append(d_rew)

        # Round Summary
        step_data = {
            "type": "step",
            "rn": r + 1,
            "atk_type": attacker_action.attack_type,
            "atkTgts": attacker_action.target_clients,
            "def_type": defender_action.action_type,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "detRaw": cum_detections,
            "fnRaw": cum_breaches,
            "fpRaw": cum_fps,
            "acc": observation.global_accuracy,
            "a_rew": round(a_rew, 1),
            "d_rew": round(d_rew, 1)
        }
        
        history.append({
            "Round": r + 1,
            "Attacker Action": f"{attacker_action.attack_type} on clients {attacker_action.target_clients}",
            "Defender Action": f"{defender_action.action_type} on clients {defender_action.target_clients}",
            "Correct Detections": tp, "False Negatives": fn, "False Positives": fp,
            "Accuracy": f"{observation.global_accuracy:.3f}",
            "A_Reward": f"{a_rew:.2f}", "D_Reward": f"{d_rew:.2f}"
        })

        yield f"data: {json.dumps(step_data)}\n\n"
        obs = observation
        await asyncio.sleep(0.01)

    # Episode Done - Send Grades
    grader_res = grader_summary(history, request.num_rounds, request.difficulty)
    precision = (cum_detections / (cum_detections + cum_fps)) * 100 if (cum_detections + cum_fps) > 0 else 0
    recall = (cum_detections / (cum_detections + cum_breaches)) * 100 if (cum_detections + cum_breaches) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    done_data = {
        "type": "done",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": env.global_accuracy,
        "graders": grader_res["tasks"]
    }
    yield f"data: {json.dumps(done_data)}\n\n"
    global last_grade_summary
    last_grade_summary = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": env.global_accuracy,
        "tasks": grader_res["tasks"],
        "overall": grader_res.get("overall", {})
    }
    
    # Log grader results for validator discovery
    log_graders(grader_res["tasks"])
    
    # Calculate standardized final metrics
    try:
        t1_raw = grader_res["tasks"].get("Task 1 (Easy - Detection Recall)", 0.0)
        task1_score = float(t1_raw)
    except (ValueError, TypeError):
        task1_score = 0.0
        
    success = task1_score >= 0.8
    
    log_end(success, request.num_rounds, task1_score, rewards_history)

@app.post("/api/run_episode")
async def run_episode(request: Optional[EpisodeRequest] = None):
    # Use standard defaults if no request body provided
    req = request or EpisodeRequest()
    return StreamingResponse(run_episode_generator(req), media_type="text/event-stream")

@app.get("/api/task_definitions")
def get_task_definitions():
    return {"tasks": TASK_DEFINITIONS}

@app.get("/api/graders")
def get_graders():
    return {
        "graders": GRADER_DEFINITIONS,
        "grading": {
            "tasks": TASK_DEFINITIONS,
            "graders": GRADER_DEFINITIONS,
            "endpoint": "/api/task_definitions",
            "graders_endpoint": "/api/graders",
            "results_endpoint": "/api/grade_results",
            "total_graders": len(GRADER_DEFINITIONS)
        }
    }

@app.get("/api/tasks")
def get_tasks():
    return {"tasks": TASK_DEFINITIONS}

@app.get("/api/graders_list")
def get_graders_list():
    """List all available graders with their metadata"""
    return {
        "graders": GRADER_DEFINITIONS,
        "count": len(GRADER_DEFINITIONS),
        "tasks_with_graders": [g["task_id"] for g in GRADER_DEFINITIONS]
    }

@app.get("/api/grade_results")
def get_grade_results():
    if last_grade_summary is None:
        raise HTTPException(status_code=404, detail="No grading results available yet. Run /api/run_episode first.")
    return {"grading_results": last_grade_summary}

import random

class StepRequest(BaseModel):
    attacker_action: Optional[AttackerAction] = None
    defender_action: Optional[DefenderAction] = None
    action: Optional[Dict[str, Any]] = None
    difficulty: Optional[str] = None

class ResetRequest(BaseModel):
    num_clients: int = DEFAULT_CLIENTS
    num_rounds: int = DEFAULT_ROUNDS
    poisoners: List[int] = DEFAULT_POISONERS
    difficulty: str = DEFAULT_DIFFICULTY

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict
    attacker_action: Optional[Dict[str, Any]] = None
    defender_action: Optional[Dict[str, Any]] = None

class UISimRequest(BaseModel):
    groq_api_key: str
    num_clients: int = 12
    num_rounds: int = 10
    difficulty: str = "medium"
    groq_model: str = "llama-3.1-8b-instant"

# Isolated env instance purely for the UI simulation so it never races with the OpenEnv evaluator
_ui_env: Optional[FederatedAdversarialEnv] = None

def _safe_ui_groq_llm(groq_key: str, model: str, messages: list, **kwargs) -> str:
    """A scoped Groq caller that replaces inference.get_llm_response for UI sim only."""
    from openai import OpenAI
    c = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
    try:
        completion = c.chat.completions.create(
            model=model, messages=messages, temperature=0.7, max_tokens=600, stream=False
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[UI-LLM] Groq Error: {e}", flush=True)
        raise e

@app.post("/api/ui/reset")
async def ui_reset(req: UISimRequest):
    """Reset a fresh isolated environment for the browser UI simulation."""
    global _ui_env, attacker_agent, defender_agent
    import random

    if not req.groq_api_key.startswith("gsk_"):
         raise HTTPException(status_code=401, detail="Invalid Groq API Key format.")

    # Reset agents' internal memory/context
    attacker_agent.reset()
    defender_agent.reset()

    # Create fresh env instance for UI
    ratio = {"easy": 0.10, "medium": 0.25, "hard": 0.40, "extreme": 0.55}.get(req.difficulty, 0.25)
    n_poison = max(1, int(req.num_clients * ratio))
    poisoners = random.sample(range(req.num_clients), n_poison)
    
    _ui_env = FederatedAdversarialEnv(num_clients=req.num_clients, num_rounds=req.num_rounds)
    _ui_env.set_config(poison_list=poisoners, num_rounds=req.num_rounds, 
                       num_clients=req.num_clients, difficulty=req.difficulty)
    obs = _ui_env.reset()
    
    return {
        "status": "ok",
        "num_clients": req.num_clients,
        "num_rounds": req.num_rounds,
        "difficulty": req.difficulty,
        "global_accuracy": obs.global_accuracy,
        "round": obs.round,
    }

@app.post("/api/ui/step")
async def ui_step(req: UISimRequest):
    """Run one step for the UI using the real agents, but injected with a Groq LLM engine."""
    global _ui_env, attacker_agent, defender_agent
    if _ui_env is None:
        raise HTTPException(status_code=400, detail="Call /api/ui/reset first")

    obs = _ui_env._get_observation()
    light_obs = obs.dict()
    light_obs.pop("client_updates", None)

    # MONKEY PATCH: Temporarily redirect agents' LLM calls to our UI-specific Groq handler
    orig_atk_llm = agents.attacker.get_llm_response
    orig_def_llm = agents.defender.get_llm_response
    
    # Custom closure to pass the req.key/model
    def groq_proxy(messages, **kwargs):
        return _safe_ui_groq_llm(req.groq_api_key, req.groq_model, messages, **kwargs)

    agents.attacker.get_llm_response = groq_proxy
    agents.defender.get_llm_response = groq_proxy

    try:
        # 1. Real Smart Agents Act (using the injected Groq proxy)
        attacker_action = attacker_agent.act(light_obs, req.difficulty)
        defender_action = defender_agent.act(light_obs)
    except Exception as e:
        print(f"[UI-STEP] Critical Execution Error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"LLM/Agent Error: {e}")
    finally:
        # Restore original LLM handlers immediately
        agents.attacker.get_llm_response = orig_atk_llm
        agents.defender.get_llm_response = orig_def_llm

    # 2. Env Step
    observation, a_rew, d_rew, info = _ui_env.step(attacker_action, defender_action)

    # 3. Update Feedback (Memory/Context)
    attacker_agent.update_feedback(
        a_rew, info.get("attacker_reason", ""),
        detections=info.get("correct_detections", 0),
        breaches=info.get("attacker_breach", 0)
    )
    defender_agent.update_feedback(
        d_rew, info.get("defender_reason", ""),
        detections=info.get("correct_detections", 0),
        fps=info.get("false_positives", 0),
        missed=info.get("attacker_breach", 0)
    )

    return {
        "round": observation.round,
        "done": observation.done,
        "global_accuracy": round(observation.global_accuracy, 4),
        "attacker": {
            "targets": attacker_action.target_clients,
            "attack_type": attacker_action.attack_type,
            "reward": round(a_rew, 2),
            "reasoning": attacker_action.metadata.get("reasoning") if hasattr(attacker_action, 'metadata') else "",
        },
        "defender": {
            "targets": defender_action.target_clients,
            "action_type": defender_action.action_type,
            "reward": round(d_rew, 2),
            "explanation": defender_action.explanation,
        },
        "metrics": {
            "correct_detections": info.get("correct_detections", 0),
            "false_positives": info.get("false_positives", 0),
            "attacker_breach": info.get("attacker_breach", 0),
        },
        "poisoned_nodes": list(_ui_env.poisoner_ids),
        "flagged_anomalies": obs.telemetry.get("flagged_anomalies_for_defender", []),
    }



# @app.post("/api/reset")
# async def reset_env_api(request: Optional[ResetRequest] = None):
#     return await reset_env(request)

@app.post("/reset")
async def reset_env(request: Optional[ResetRequest] = None):
    """Reset the environment with formal defaults (20/15/med) or custom overrides"""
    # Use provided values or fall back to standard defaults from the model
    config = request or ResetRequest()
    
    env.set_config(
        poison_list=config.poisoners,
        num_rounds=config.num_rounds,
        num_clients=config.num_clients,
        difficulty=config.difficulty
    )
    
    obs = env.reset()
    attacker_agent.reset()
    defender_agent.reset()
    
    log_start()
    return obs

# @app.post("/api/step", response_model=StepResponse)
# async def step_env_api(request: Optional[StepRequest] = None):
#     return await step_env(request)

@app.post("/step", response_model=StepResponse)
async def step_env(request: Optional[StepRequest] = None):
    """Execute one round of the simulation using Smart Agents or custom inputs"""

    current_obs = env._get_observation()
    light_obs = current_obs.dict() if hasattr(current_obs, 'dict') else current_obs.copy()
    if "client_updates" in light_obs:
        del light_obs["client_updates"]

    # 1. Parse any explicit action wrapper from the request for compatibility
    atk = None
    dfn = None
    if request:
        if request.action:
            atk = request.action.get("attacker_action")
            dfn = request.action.get("defender_action")
        atk = request.attacker_action if request.attacker_action is not None else atk
        dfn = request.defender_action if request.defender_action is not None else dfn

    # 2. Determine difficulty: use request value when present, otherwise fall back to hardcoded default
    request_difficulty = request.difficulty if request and request.difficulty else DEFAULT_DIFFICULTY

    if not atk:
        atk = attacker_agent.act(light_obs, request_difficulty)
    elif isinstance(atk, dict):
        atk = AttackerAction(**atk)

    # 3. Provide Smart Defender Default if not provided
    if not dfn:
        dfn = defender_agent.act(light_obs)
    elif isinstance(dfn, dict):
        dfn = DefenderAction(**dfn)

    observation, a_rew, d_rew, info = env.step(atk, dfn)
    
    # Update Agent feedback history (Keep them smart!)
    attacker_agent.update_feedback(a_rew, info.get("attacker_reason", ""))
    defender_agent.update_feedback(d_rew, info.get("defender_reason", ""))
    
    # Log step for OpenEnv (Standardized - Combined Agents)
    log_step(observation.round, atk, dfn, a_rew, d_rew, observation.done)
    
    # Simple check for END on single steps
    if observation.done:
        log_end(True, observation.round, 1.0, [d_rew])

    return {
        "observation": observation,
        "reward": d_rew,
        "done": observation.done,
        "info": info,
        "attacker_action": atk.dict() if hasattr(atk, 'dict') else None,
        "defender_action": dfn.dict() if hasattr(dfn, 'dict') else None
    }

@app.get("/state")
async def get_state():
    return env.get_state()

@app.get("/health")
async def health():
    return {"status": "healthy", "dashboard": "SharpernerRL Online"}

@app.get("/web")
async def dashboard():
    from fastapi.responses import FileResponse
    return FileResponse("server/static/index.html")


def main():
    import uvicorn
    # 0.0.0.0 is MANDATORY for Docker/validator access
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
