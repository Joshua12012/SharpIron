from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import asyncio
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from environment import FederatedAdversarialEnv
from models import AttackerAction, DefenderAction, Observation, RewardInfo
from attacker import AttackerAgent
from defender import DefenderAgent
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
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global instances
env = FederatedAdversarialEnv(num_clients=20, num_rounds=15)
attacker_agent = AttackerAgent()
defender_agent = DefenderAgent()

class EpisodeRequest(BaseModel):
    poisoners: list
    num_clients: int
    num_rounds: int
    difficulty: str

async def run_episode_generator(request: EpisodeRequest):
    """Generator for streaming round-by-round results to the UI"""
    env.num_clients = request.num_clients
    env.num_rounds = request.num_rounds
    env.set_config(request.poisoners, request.num_rounds, difficulty=request.difficulty)
    
    obs = env.reset()
    attacker_agent.reset()
    defender_agent.reset()
    
    history = []
    cum_detections = 0
    cum_breaches = 0
    cum_fps = 0

    yield f"data: {json.dumps({'type': 'init', 'msg': 'Simulation started'})}\n\n"

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

@app.post("/api/run_episode")
async def run_episode(request: EpisodeRequest):
    return StreamingResponse(run_episode_generator(request), media_type="text/event-stream")

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy", "dashboard": "SENTINEL 2.3 Online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)