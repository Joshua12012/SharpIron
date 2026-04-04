# env.py
"""
Core Environment for Red-Blue Adversarial Federated Learning
Manages global model simulation, telemetry, and episode state.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from models import Observation, AttackerAction, DefenderAction
from rewards import calculate_attacker_reward, calculate_defender_reward


class FederatedAdversarialEnv:
    def __init__(self, num_clients: int = 20, num_rounds: int = 15):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.current_round = 0
        self.global_accuracy = 0.85  # Starting accuracy
        self.client_updates = None
        self.poisoner_ids = []
        self.done = False
        self.difficulty = "medium"   # default
        # History for pattern detection
        self.update_history = []  # List of previous client update norms
        
        self.reset()

    def reset(self) -> Observation:
        """Reset episode"""
        self.current_round = 0
        self.global_accuracy = 0.85
        self.update_history = []
        self.done = False
        self.poisoner_ids = []
        
        # Generate initial benign updates
        self.client_updates = self._generate_client_updates(is_poisoned=False)
        
        return self._get_observation()

    def _generate_client_updates(self, is_poisoned: bool = False, attack_strength: float = 1.0):
        """Generate simulated client weight updates"""
        updates = []
        for i in range(self.num_clients):
            # FIX: Previously, if is_poisoned was True, ALL 20 clients were receiving massive poisoned updates.
            # Now, it correctly applies poisoned noise ONLY to the selected poisoner_ids,
            # meaning the telemetry will actually highlight the correct clients instead of being random static.
            if is_poisoned and getattr(self, 'poisoner_ids', None) and i in self.poisoner_ids:
                # Poisoned update - scaled and noisy
                base = np.random.normal(0, 0.5, 10) * attack_strength
                updates.append(base.tolist())
            else:
                # Benign update - small natural variation
                base = np.random.normal(0, 0.1, 10)
                updates.append(base.tolist())
        return updates

    # def step(self, attacker_action: AttackerAction, defender_action: DefenderAction) -> Tuple[Observation, float, float, Dict]:
    #     """One step: Attacker acts → Environment updates → Defender acts → Rewards"""
    #     self.current_round += 1
        
    #     # 1. Attacker executes attack
    #     self.poisoner_ids = attacker_action.target_clients
    #     attack_strength = attacker_action.strength
    #     stealth = attacker_action.stealth_level
        
    #     # Generate updates based on attacker choice
    #     self.client_updates = self._generate_client_updates(
    #         is_poisoned=True if self.poisoner_ids else False,
    #         attack_strength=attack_strength * (1.0 - stealth * 0.4)  # stealth reduces obviousness
    #     )
        
    #     # 2. Simulate global model update and accuracy impact
    #     previous_acc = self.global_accuracy
    #     self._update_global_model()
        
    #     accuracy_drop = previous_acc - self.global_accuracy
        
    #     # 3. Prepare success metrics for rewards (without revealing ground truth to agents)
    #     success_metrics = {
    #         "accuracy_drop": accuracy_drop,
    #         "single_node_success": len(self.poisoner_ids) == 1 and accuracy_drop > 0.02,
    #         "coordinated_success": len(self.poisoner_ids) > 1 and accuracy_drop > 0.03,
    #         "alie_success": len(self.poisoner_ids) > 0 and accuracy_drop > 0.015,  # A Little Is Enough
    #         "stealth_success": stealth > 0.6 and accuracy_drop > 0.01,
    #         "pattern_detected": defender_action.action_type in ["detect", "investigate"],
    #         "coordinated_detected": len(defender_action.target_clients) > 1,
    #         "single_detected": len(defender_action.target_clients) == 1,
    #         "multi_device_detected": len(defender_action.target_clients) > 2,
    #         "accuracy_maintained": self.global_accuracy > previous_acc - 0.02
    #     }
        
    #     # 4. Calculate rewards
    #     attacker_reward, attacker_reason = calculate_attacker_reward(attacker_action, success_metrics)
    #     defender_reward, defender_reason = calculate_defender_reward(defender_action, success_metrics)
        
    #     # 5. Check if episode is done
    #     self.done = self.current_round >= self.num_rounds
        
    #     obs = self._get_observation()
        
    #     info = {
    #         "round": self.current_round,
    #         "attacker_reason": attacker_reason,
    #         "defender_reason": defender_reason,
    #         "accuracy_drop": accuracy_drop
    #     }
        
    #     return obs, attacker_reward, defender_reward, info

    def step(self, attacker_action: AttackerAction, defender_action: DefenderAction):
        self.current_round += 1
        
        # Attacker executes
        self.poisoner_ids = attacker_action.target_clients
        
        # Check if attacker is in imitation mode (new behavior)
        is_imitation = attacker_action.attack_type == "imitation"
        
        self.client_updates = self._generate_client_updates(
            is_poisoned=bool(self.poisoner_ids) and not is_imitation,
            attack_strength=attacker_action.strength
        )
        
        previous_acc = self.global_accuracy
        # FIX: Now explicitly passing the attacker's strength to the global model update
        self._update_global_model(attack_strength=attacker_action.strength)
        
        accuracy_drop = previous_acc - self.global_accuracy
        
        # # Success metrics (no direct ground truth to agents)
        # success_metrics = {
        #     "accuracy_drop": accuracy_drop,
        #     "single_node_success": len(self.poisoner_ids) == 1 and accuracy_drop > 0.015,
        #     "coordinated_success": len(self.poisoner_ids) > 1 and accuracy_drop > 0.025,
        #     "alie_success": len(self.poisoner_ids) > 0 and 0.01 < accuracy_drop < 0.04,
        #     "stealth_success": attacker_action.stealth_level > 0.7 and accuracy_drop > 0.01,
        #     "imitation_success": is_imitation and accuracy_drop < 0.005,
        #     "pattern_detected": defender_action.action_type in ["detect", "investigate"],
        #     "coordinated_detected": len(defender_action.target_clients) > 1,
        #     "single_detected": len(defender_action.target_clients) == 1,
        #     "detection_round": self.current_round,
        #     "accuracy_maintained": accuracy_drop < 0.03
        # }
        
        # Calculate accurate success metrics for rewards
        real_poisoned = set(self.poisoner_ids)
        defender_flagged = set(defender_action.target_clients)
        
        correct_detections = len(real_poisoned & defender_flagged)        # True positives
        false_positives = len(defender_flagged - real_poisoned)           # False positives
        
        success_metrics = {
            "accuracy_drop": accuracy_drop,
            "correct_detections": correct_detections,
            "false_positives": false_positives,
            "coordinated_detected": len(defender_flagged) > 1,
            "single_detected": len(defender_flagged) == 1,
            "pattern_detected": defender_action.action_type in ["detect", "investigate"],
            "detection_round": self.current_round,
            "accuracy_maintained": accuracy_drop < 0.015,
            "stealth_success": attacker_action.stealth_level > 0.7 and accuracy_drop > 0.01,
            "coordinated_success": len(self.poisoner_ids) > 1 and accuracy_drop > 0.02,
            "alie_success": len(self.poisoner_ids) > 0 and 0.01 < accuracy_drop < 0.04,
            "imitation_success": getattr(attacker_action, 'attack_type', '') == "imitation"
        }
        
        attacker_reward, attacker_reason = calculate_attacker_reward(attacker_action, success_metrics, self.current_round)
        defender_reward, defender_reason = calculate_defender_reward(defender_action, success_metrics, self.current_round)
        
        self.done = self.current_round >= self.num_rounds
        
        obs = self._get_observation()
        
        info = {
            "round": self.current_round,
            "attacker_reason": attacker_reason,
            "defender_reason": defender_reason,
            "accuracy_drop": accuracy_drop
        }
        
        return obs, attacker_reward, defender_reward, info

    def _update_global_model(self, attack_strength: float = 1.0):
        """Simple simulation of global model update"""
        # In real FL this would be aggregation. Here we simulate impact
        if self.poisoner_ids:
            # FIX: Multiplier bumped to 0.25 so the model accuracy is significantly reduced by sustained attacks.
            # Removed the artificial 0.45 floor so accuracy can genuinely drop to 0 in extended rounds.
            damage = 0.25 * attack_strength * len(self.poisoner_ids) / self.num_clients
            self.global_accuracy = max(0.0, self.global_accuracy - damage)
        else:
            self.global_accuracy = min(0.96, self.global_accuracy + 0.01)

    def _get_observation(self) -> Observation:
        """Observation visible to both agents (no ground truth poisoner count)"""
        # Compute some telemetry features
        norms = [float(np.linalg.norm(np.array(update))) for update in self.client_updates]
        variances = [float(np.var(update)) for update in self.client_updates]
        
        # FIX: The Defender LLM was struggling to associate large raw arrays with the correct client index,
        # causing it to randomly guess [0, 1, 13] constantly. 
        # Here we pre-digest the arrays to highlight statistical outliers (top 3 highest norms) to guide the defender.
        anomalies = sorted(range(len(norms)), key=lambda i: norms[i], reverse=True)[:3]
        
        telemetry = {
            "update_norms": [round(n, 4) for n in norms],
            "update_variances": [round(v, 4) for v in variances],
            "flagged_anomalies_for_defender": anomalies,
            "global_accuracy": self.global_accuracy,
            "total_clients": self.num_clients,
            "round": self.current_round
        }
        
        return Observation(
            client_updates=self.client_updates,
            global_accuracy=self.global_accuracy,
            round=self.current_round,
            telemetry=telemetry,
            done=self.done
        )

    def get_state(self):
        """For debugging / logging"""
        return {
            "round": self.current_round,
            "global_accuracy": self.global_accuracy,
            "poisoner_count": len(self.poisoner_ids),
            "done": self.done
        }
    
    def set_config(self, poison_list: list, num_rounds: int, difficulty: str = "medium"):
        """Update environment configuration including difficulty level"""
        self.num_rounds = int(num_rounds)
        self.difficulty = difficulty.lower()
        self.poisoner_ids = sorted(poison_list) if poison_list else []
        
        # Reset episode state
        self.current_round = 0
        self.global_accuracy = 0.85
        self.update_history = []
        self.done = False
        
        # Generate initial client updates based on difficulty
        self.client_updates = self._generate_client_updates(
            is_poisoned=bool(self.poisoner_ids),
            attack_strength=1.0
        )
        
        print(f"[Env] Config updated → Difficulty: {self.difficulty}, "
              f"Poisoners: {self.poisoner_ids}, Clients: {self.num_clients}, Rounds: {self.num_rounds}")