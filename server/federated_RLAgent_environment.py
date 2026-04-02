# env.py
"""
Federated Poisoning Detector - OpenEnv Style
Updated with stronger adaptive attacker and debugging.
"""

import numpy as np
import random
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import FederatedPoisoningAction, FederatedPoisoningObservation
from datetime import datetime


# Global counter for debugging API calls (even if we don't call real LLM yet)
API_CALL_COUNT = 0

def log_debug(message: str):
    """Simple debug logger to track what happens during simulation."""
    global API_CALL_COUNT
    API_CALL_COUNT += 1
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] DEBUG #{API_CALL_COUNT}: {message}")


class FederatedPoisoningEnv(Environment):
    """Main OpenEnv environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, num_clients: int = 10, num_rounds: int = 8):
        super().__init__()
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self._state = State(episode_id="", step_count=0)
        self.reset()

    def reset(self) -> FederatedPoisoningObservation:
        """Reset the environment."""
        self._state = State(episode_id=str(random.randint(100000, 999999)), step_count=0)
        
        self.round = 0
        self.detected = set()
        self.previous_acc = 0.52
        self.current_acc = 0.52
        self.total_reward = 0.0
        self.suspicion = np.zeros(self.num_clients)
        
        # self.poisoner_ids = [2, 5, 7]   # Change this to test different setups
        if not hasattr(self, "poisoner_ids"):
            self.poisoner_ids = [2, 5, 7]
        
        log_debug(f"Reset episode. Poisoners: {self.poisoner_ids}")
        
        self.client_stats = self._generate_stats()
        
        obs = self._get_obs()
        
        return FederatedPoisoningObservation(
            observation=obs.tolist(),
            round=self.round,
            accuracy=self.current_acc,
            done=False,
            reward=0.0,
            metadata={"poisoner_ids": self.poisoner_ids.copy()}
        )

    def set_config(self, poison_list, num_rounds):
        """Update from Gradio UI."""
        
        self.num_rounds = int(num_rounds)

        # Keep only poisoners that are within valid client range
        self.poisoner_ids = sorted([
            int(i) for i in poison_list
            if int(i) < self.num_clients
        ])

        self.round = 0
        self.detected = set()
        self.previous_acc = 0.52
        self.current_acc = 0.52
        self.total_reward = 0.0

        # Resize suspicion BEFORE generating stats
        self.suspicion = np.zeros(self.num_clients)

        # Now safe
        self.client_stats = self._generate_stats()

        log_debug(
            f"Config updated - Clients: {self.num_clients}, "
            f"Poisoners: {self.poisoner_ids}, Rounds: {self.num_rounds}"
        )

        return f"Updated: {len(self.poisoner_ids)} poisoners, {self.num_rounds} rounds"
    def _generate_stats(self):
        """Improved adaptive attacker - each poisoned client adapts independently."""
        stats = np.zeros((self.num_clients, 5))
        
        # Count how many poisoned clients are heavily flagged
        flagged_count = sum(1 for i in self.poisoner_ids if self.suspicion[i] > 1.5)
        
        for i in range(self.num_clients):
            if i not in self.poisoner_ids:
                # Benign clients - stable behavior
                stats[i, 0] = np.random.uniform(0.75, 2.55)   # norm
                stats[i, 1] = np.random.uniform(0.81, 0.96)   # similarity
                stats[i, 2] = np.random.uniform(0.08, 0.78)   # variance
                stats[i, 3] = np.random.uniform(-0.16, 0.16)  # loss delta
                stats[i, 4] = np.random.uniform(0.05, 0.42)   # anomaly
            else:
                # Adaptive poisoned client
                flags = self.suspicion[i]
                
                if flags < 0.8:
                    # Ignored → become aggressive
                    strength = 3.2 + 1.1 * (self.round / max(1, self.num_rounds))
                elif flags < 2.2:
                    # Moderately suspected → medium attack
                    strength = 1.85
                else:
                    # Heavily suspected → very stealthy
                    strength = 1.06
                
                # Coordination: if many poisoned clients are flagged, all become stealthier
                coord_factor = 0.25 * (flagged_count / max(1, len(self.poisoner_ids)))
                strength = strength * (1.0 - coord_factor * 0.6)
                
                stats[i, 0] = np.random.uniform(3.2, 8.2) * strength          # norm
                stats[i, 1] = np.random.uniform(0.18, 0.68)                    # similarity
                stats[i, 2] = np.random.uniform(1.4, 4.8)                      # variance
                stats[i, 3] = np.random.uniform(0.28, 1.35)                    # loss delta
                stats[i, 4] = np.random.uniform(0.55, 2.1)                     # anomaly
                
                log_debug(f"Client {i} (poisoned) - Flags: {flags:.1f}, Strength: {strength:.2f}")
        
        return stats.astype(np.float32)

    def _get_obs(self):
        obs = self.client_stats.copy()
        round_progress = np.full((self.num_clients, 1), self.round / max(1, self.num_rounds))
        suspicion_col = self.suspicion.reshape(-1, 1)
        return np.concatenate([obs, round_progress, suspicion_col], axis=1).astype(np.float32)

    def step(self, action: FederatedPoisoningAction) -> FederatedPoisoningObservation:
        self.round += 1
        action_list = np.array(action.action_list).flatten()
        
        correct_dets = 0
        false_pos = 0
        
        for i in range(self.num_clients):
            dec = int(action_list[i])
            is_poison = i in self.poisoner_ids
            
            if is_poison and dec >= 1:
                correct_dets += 1
                self.detected.add(i)
                self.suspicion[i] += 1.25
            elif not is_poison and dec >= 1:
                false_pos += 1
                self.suspicion[i] = max(0, self.suspicion[i] - 0.9)
            
            self.suspicion[i] = max(0, self.suspicion[i] * 0.82)
        
        self.previous_acc = self.current_acc
        acc_improvement = 0.07 * correct_dets - 0.04 * false_pos
        self.current_acc = min(0.96, self.previous_acc + acc_improvement)
        
        reward = self._compute_reward(correct_dets, false_pos, acc_improvement)
        self.total_reward += reward
        
        done = self.round >= self.num_rounds
        obs = self._get_obs()
        
        info = {
            "round": self.round,
            "accuracy": self.current_acc,
            "correct_dets": correct_dets,
            "false_pos": false_pos,
            "detected": len(self.detected)
        }
        
        return FederatedPoisoningObservation(
            observation=obs.tolist(),
            round=self.round,
            accuracy=self.current_acc,
            done=done,
            reward=reward,
            metadata=info
        )

    # def _compute_reward(self, correct_dets, false_pos, acc_improvement):
    #     """Rich reward function"""
    #     reward = 6.0 * correct_dets - 3.8 * false_pos + 28.0 * acc_improvement
        
    #     if self.round == self.num_rounds:
    #         recall = len(self.detected) / max(1, len(self.poisoner_ids))
    #         precision = len(self.detected) / max(1, len(self.detected) + false_pos)
    #         f1 = 2 * precision * recall / (precision + recall + 1e-8)
    #         reward += 52.0 * f1 + 38.0 * self.current_acc
        
    #     reward = np.clip(reward, -15.0, 25.0)
    #     return float(reward)
    def _compute_reward(self, correct_dets, false_pos, acc_improvement):
        """Improved reward function - encourages catching poisoned clients
        while penalizing uncaptured poison and false positives.
        Agent does NOT see the true number of poisoned clients.
        """
        num_poisoned = len(self.poisoner_ids)
        uncaptured = num_poisoned - correct_dets
        
        # Main reward components
        detection_bonus = 5.5 * correct_dets
        false_positive_penalty = -4.2 * false_pos
        uncaptured_penalty = -7.8 * uncaptured          # Stronger penalty for missed poison
        
        # Bonus for maintaining accuracy
        accuracy_bonus = 22.0 * acc_improvement
        
        reward = detection_bonus + false_positive_penalty + uncaptured_penalty + accuracy_bonus
        
        # Final round bonus (encourages good overall performance)
        if self.round == self.num_rounds:
            recall = correct_dets / max(1, num_poisoned)
            precision = correct_dets / max(1, correct_dets + false_pos)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            reward += 45.0 * f1 + 30.0 * self.current_acc
        
        # Clip to prevent extreme values
        reward = np.clip(reward, -25.0, 30.0)
        
        return float(reward)
    @property
    def state(self) -> State:
        return self._state