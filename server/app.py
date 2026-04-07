from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import asyncio
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from .environment import FederatedAdversarialEnv
from models import AttackerAction, DefenderAction, Observation, RewardInfo
from agents.attacker import AttackerAgent
from agents.defender import DefenderAgent
from graders import grader_summary

app = FastAPI(title="SharpernerRL")

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

# Standard Defaults for OpenEnv Validation
DEFAULT_CLIENTS = 20
DEFAULT_ROUNDS = 15
DEFAULT_POISONERS = [3, 7, 12]
DEFAULT_DIFFICULTY = "medium"

# Global instances
env = FederatedAdversarialEnv(num_clients=DEFAULT_CLIENTS, num_rounds=DEFAULT_ROUNDS)
attacker_agent = AttackerAgent()
defender_agent = DefenderAgent()

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

import random

class StepRequest(BaseModel):
    attacker_action: Optional[AttackerAction] = None
    defender_action: Optional[DefenderAction] = None

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

@app.post("/step", response_model=StepResponse)
async def step_env(request: Optional[StepRequest] = None):
    """Execute one round of the simulation using Smart Agents or custom inputs"""
    
    # Get current telemetry for the agents to reason about
    current_obs = env._get_observation()
    light_obs = current_obs.dict() if hasattr(current_obs, 'dict') else current_obs.copy()
    if "client_updates" in light_obs:
        del light_obs["client_updates"]

    # 1. Provide Smart Attacker Default (if not provided)
    atk = request.attacker_action if request and request.attacker_action else None
    if not atk:
        atk = attacker_agent.act(light_obs, DEFAULT_DIFFICULTY)
    
    # 2. Provide Smart Defender Default (if not provided)
    dfn = request.defender_action if request and request.defender_action else None
    if not dfn:
        dfn = defender_agent.act(light_obs)

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
        "info": info
    }

@app.get("/state")
async def get_state():
    return env.get_state()

@app.get("/health")
async def health():
    return {"status": "healthy", "dashboard": "SharpernerRL Online"}

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("server/static/index.html")


def main():
    import uvicorn
    # 0.0.0.0 is MANDATORY for Docker/validator access
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
