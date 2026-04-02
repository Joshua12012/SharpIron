# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Federated Rlagent Environment.

The federated_RLAgent environment is a simple test environment that echoes back messages.
"""
# models.py
"""
Data models for Federated Poisoning Detector Environment.
Defines the Action and Observation that the RL agent will use.
"""

from typing import List, Dict, Any
from pydantic import BaseModel


class FederatedPoisoningAction(BaseModel):
    """
    Action taken by the RL agent.
    For each client, the agent chooses:
      0 = Keep (use the update)
      1 = Flag (reduce weight, mark suspicious)
      2 = Quarantine (completely remove this round)
    """
    action_list: List[int]   # length = num_clients


class FederatedPoisoningObservation(BaseModel):
    """
    Observation returned to the RL agent after each step.
    Contains statistical features for each client.
    """
    observation: List[List[float]]   # shape: (num_clients, 6)
    round: int
    accuracy: float
    done: bool
    reward: float
    metadata: Dict[str, Any] = {}