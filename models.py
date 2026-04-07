# models.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AttackerAction(BaseModel):
    """Actions the Attacker (Red) can take"""
    target_clients: List[int]          # Which clients to attack
    attack_type: str                   # "single", "coordinated", "alie", "stealth"
    strength: float                    # 0.0 to 1.0
    stealth_level: float = 0.5         # How hidden the attack is
    metadata: Optional[Dict] = None


class DefenderAction(BaseModel):
    """Actions the Defender (Blue) can take"""
    action_type: str                   # "detect", "quarantine", "reweight", "ignore", "investigate"
    target_clients: List[int]
    confidence: float                  # 0.0 to 1.0
    explanation: Optional[str] = None


class Observation(BaseModel):
    """What both agents observe"""
    client_updates: List[List[float]]  # Simulated weight updates (simplified)
    global_accuracy: float
    round: int
    telemetry: Dict[str, Any]          # norms, variances, etc.
    done: bool


class RewardInfo(BaseModel):
    attacker_reward: float
    defender_reward: float
    attacker_reason: str
    defender_reason: str