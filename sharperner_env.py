import httpx
import os
import asyncio
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class SharpernerAction(BaseModel):
    """Container for both attacker and defender actions as expected by the backend"""
    attacker_action: Dict[str, Any]
    defender_action: Dict[str, Any]

class SharpernerObservation(BaseModel):
    client_updates: Optional[List[List[float]]] = None
    global_accuracy: float
    round: int
    telemetry: Dict[str, Any]
    done: bool

class SharpernerStepResult:
    def __init__(self, observation: SharpernerObservation, reward: float, done: bool, info: Dict[str, Any]):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

class SharpernerEnv:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None):
        """
        In a real OpenEnv setup, this might start a container.
        For this submission, we assume the backend is already running (e.g. via Docker Compose or HF Space).
        """
        # If running in a validator, the base URL might be provided via env var
        url = os.getenv("PING_URL", "http://localhost:8000")
        return cls(base_url=url)

    async def reset(self, config: Optional[Dict[str, Any]] = None) -> SharpernerStepResult:
        """Calls POST /reset"""
        payload = config if config else {}
        response = await self.client.post(f"{self.base_url}/reset", json=payload)
        response.raise_for_status()
        obs_data = response.json()
        obs = SharpernerObservation(**obs_data)
        return SharpernerStepResult(obs, 0.0, obs.done, {})

    async def step(self, action: SharpernerAction) -> SharpernerStepResult:
        """Calls POST /step"""
        response = await self.client.post(f"{self.base_url}/step", json=action.dict())
        response.raise_for_status()
        data = response.json()
        obs = SharpernerObservation(**data["observation"])
        return SharpernerStepResult(
            observation=obs,
            reward=data["reward"],
            done=data["done"],
            info=data["info"]
        )

    async def close(self):
        await self.client.aclose()
