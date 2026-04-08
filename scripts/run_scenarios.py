import asyncio
import os
import json
import sys
from pathlib import Path
# Ensure the absolute root directory of the project is in the Python path
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sharperner_env import SharpernerEnv, SharpernerAction

async def run_scenario(name, poisoners, difficulty, atk_behavior, def_behavior):
    print(f"\n===== SCENARIO: {name} =====")
    print(f"Configs: Poisoners={poisoners}, Difficulty={difficulty}")
    
    base_url = os.getenv("PING_URL", "http://localhost:8000")
    env = SharpernerEnv(base_url=base_url)
    
    try:
        # 1. RESET
        print("[RESET] Initializing scenario...")
        await env.reset({
            "num_clients": 20,
            "num_rounds": 5,
            "poisoners": poisoners,
            "difficulty": difficulty
        })
        
        # 2. RUN STEPS
        total_d_rew = 0
        for r in range(1, 6):
            # Hardcoded actions based on behavior
            atk_action = atk_behavior(r)
            def_action = def_behavior(r)
            
            action = SharpernerAction(
                attacker_action=atk_action,
                defender_action=def_action
            )
            
            result = await env.step(action)
            total_d_rew += result.reward
            
            print(f"Round {r}: Def Targets={def_action['target_clients']} | TP={result.info.get('correct_detections')} | FP={result.info.get('false_positives')} | Reward={result.reward:.2f}")
            
            if result.done:
                break
                
        print(f"Scenario Result: Total Defender Reward = {total_d_rew:.2f}")
        
    except Exception as e:
        print(f"Error in scenario: {e}")
    finally:
        await env.close()

# Define behaviors for testing
def passive_attacker(r):
    return {"attack_type": "coordinated", "target_clients": [3, 7]}

def blind_defender(r):
    return {"action_type": "detect", "target_clients": [], "confidence": 0.5}

def perfect_defender(r):
    # This logic assumes we know the poisoners for testing
    return {"action_type": "quarantine", "target_clients": [3, 7, 12], "confidence": 0.9}

async def main():
    # Scenario 1: Clean environment, blind defender
    await run_scenario("Baseline / Clean", [], "easy", passive_attacker, blind_defender)
    
    # Scenario 2: Attackers present, perfect defender
    await run_scenario("Perfect Defense", [3, 7, 12], "medium", passive_attacker, perfect_defender)

if __name__ == "__main__":
    asyncio.run(main())
