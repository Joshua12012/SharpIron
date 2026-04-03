# main.py
"""
FastAPI Server for Red-Blue Adversarial Federated Learning Environment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import gradio as gr

from environment import FederatedAdversarialEnv
from models import AttackerAction, DefenderAction, Observation, RewardInfo
from attacker import AttackerAgent   # We'll create this next
from defender import DefenderAgent   # We'll create this next
from app import demo   # Import the Gradio Blocks from app.py

app = FastAPI(title="Red-Blue Adversarial Federated Learning")

# CORS for future UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = FederatedAdversarialEnv(num_clients=20, num_rounds=15)

# Placeholder agents (we'll implement LLM-based versions later)
attacker_agent = AttackerAgent()
defender_agent = DefenderAgent()


class StepRequest(BaseModel):
    attacker_action: AttackerAction
    defender_action: Optional[DefenderAction] = None  # Defender can be manual or auto


@app.get("/reset")
async def reset():
    """Reset the environment and start a new episode"""
    obs = env.reset()
    return {
        "status": "success",
        "message": "Episode reset",
        "observation": obs,
        "round": 0
    }


@app.post("/step")
async def step(request: StepRequest):
    """Execute one step: Attacker acts → Environment updates → Defender acts"""
    try:
        # If defender action is not provided, let the defender agent decide
        if request.defender_action is None:
            defender_action = defender_agent.act(env.get_state())  # Placeholder for now
        else:
            defender_action = request.defender_action

        # Execute environment step
        observation, attacker_reward, defender_reward, info = env.step(
            attacker_action=request.attacker_action,
            defender_action=defender_action
        )

        return {
            "status": "success",
            "observation": observation,
            "attacker_reward": attacker_reward,
            "defender_reward": defender_reward,
            "info": info,
            "round": env.current_round,
            "done": observation.done
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current environment state (for debugging)"""
    return env.get_state()


@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Red-Blue Adversarial FL Environment is running"}



# Mount Gradio app at /gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Health check
@app.get("/")
async def root():
    return {
        "message": "Red-Blue Adversarial FL API is running",
        "gradio_ui": "/gradio",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)