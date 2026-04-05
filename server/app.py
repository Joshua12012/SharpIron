# app.py
"""
Red-Blue Adversarial Federated Learning UI
With strict [START]/[STEP]/[END] logging for evaluation + LLM prompt/performance summary
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import json
import os
import sys

from environment import FederatedAdversarialEnv
from attacker import AttackerAgent
from defender import DefenderAgent
from models import AttackerAction, DefenderAction
from graders import grader_summary

# Global instances
env = FederatedAdversarialEnv(num_clients=20, num_rounds=15)
attacker_agent = AttackerAgent()
defender_agent = DefenderAgent()

def log_start():
    print("[START] Red-Blue Adversarial Federated Learning Episode")
    print(f"[START] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()

def log_step(round_num, attacker_action, defender_action, observation, attacker_reward, defender_reward):
    print(f"[STEP] Round {round_num}")
    print(f"[STEP] Attacker Action: {attacker_action.attack_type} on clients {attacker_action.target_clients} (strength={attacker_action.strength:.2f}, stealth={attacker_action.stealth_level:.2f})")
    print(f"[STEP] Defender Action: {defender_action.action_type} on clients {defender_action.target_clients} (confidence={defender_action.confidence:.2f})")
    print(f"[STEP] Global Accuracy: {observation.global_accuracy:.4f}")
    print(f"[STEP] Attacker Reward: {attacker_reward:.2f}")
    print(f"[STEP] Defender Reward: {defender_reward:.2f}")
    sys.stdout.flush()

def log_end(final_accuracy):
    print(f"[END] Episode Finished")
    print(f"[END] Final Global Accuracy: {final_accuracy:.4f}")
    print("[END] Red-Blue Adversarial Episode Complete")
    sys.stdout.flush()

def run_full_episode(selected_poisoners, num_clients, num_rounds, difficulty):
    """Main episode runner with strict logging and optimized context feedback"""
    log_start()
    
    # RESET AGENT MEMORY for the new episode
    attacker_agent.reset()
    defender_agent.reset()

    num_clients = int(num_clients)
    num_rounds = int(num_rounds)
    difficulty = difficulty.lower()

    env.num_clients = num_clients
    env.num_rounds = num_rounds
    env.set_config(selected_poisoners, num_rounds, difficulty=difficulty)

    obs = env.reset()
    history = []
    
    # Cumulative Metrics for Plotting
    cum_detections = [0]
    cum_fns = [0]  # False Negatives (Missed)
    cum_fps = [0]

    for r in range(num_rounds):
        # Create a light version of the observation (No raw floats)
        light_obs = obs.dict() if hasattr(obs, 'dict') else obs.copy()
        if "client_updates" in light_obs:
            del light_obs["client_updates"]

        # Agents act
        attacker_action = attacker_agent.act(light_obs, difficulty)
        defender_action = defender_agent.act(light_obs)

        # Env Step
        observation, attacker_reward, defender_reward, info = env.step(attacker_action, defender_action)
        
        # Give STRUCTURED feedback to both agents (Optimized Context)
        attacker_agent.update_feedback(
            attacker_reward, 
            info.get("attacker_reason", ""),
            detections=info.get("correct_detections", 0),
            breaches=info.get("attacker_breach", 0)
        )
        defender_agent.update_feedback(
            defender_reward, 
            info.get("defender_reason", ""),
            detections=info.get("correct_detections", 0),
            fps=info.get("false_positives", 0),
            missed=info.get("attacker_breach", 0)
        )
        
        log_step(r+1, attacker_action, defender_action, observation, attacker_reward, defender_reward)

        # Update Metrics
        cum_detections.append(cum_detections[-1] + info.get("correct_detections", 0))
        cum_fns.append(cum_fns[-1] + info.get("attacker_breach", 0))
        cum_fps.append(cum_fps[-1] + info.get("false_positives", 0))

        history.append({
            "Round": r + 1,
            "Attacker Action": f"{attacker_action.attack_type} on clients {attacker_action.target_clients}",
            "Defender Action": f"{defender_action.action_type} on clients {defender_action.target_clients}",
            "Correct Detections": info.get("correct_detections", 0),
            "False Negatives": info.get("attacker_breach", 0),
            "False Positives": info.get("false_positives", 0),
            "Accuracy": f"{observation.global_accuracy:.3f}",
            "A_Reward": f"{attacker_reward:.2f}",
            "D_Reward": f"{defender_reward:.2f}"
        })

        obs = observation
        time.sleep(0.1) # Faster simulation

    df = pd.DataFrame(history)

    # Plot Performance Dashboard
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cum_detections, mode='lines+markers', name='Detections (Caught)', line=dict(color='#1D9E75', width=3)))
    fig.add_trace(go.Scatter(y=cum_fns, mode='lines+markers', name='False Negatives (Missed)', line=dict(color='#E53E3E', width=3)))
    fig.add_trace(go.Scatter(y=cum_fps, mode='lines', name='False Positives', line=dict(color='#A0AEC0', dash='dash')))
    
    fig.update_layout(
        title=f"Agent Performance Dashboard - {difficulty.capitalize()}",
        xaxis_title="Round",
        yaxis_title="Count (Cumulative)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Analytics calculation
    total_det = cum_detections[-1]
    total_fn = cum_fns[-1]
    total_fp = cum_fps[-1]
    precision = (total_det / (total_det + total_fp)) * 100 if (total_det + total_fp) > 0 else 0
    recall = (total_det / (total_det + total_fn)) * 100 if (total_det + total_fn) > 0 else 0

    result = f"""
### 🏁 Episode Summary
- **Difficulty**: {difficulty.capitalize()}
- **Defender Detections (True Positives)**: {total_det}
- **Missed Attacks (False Negatives)**: {total_fn} (Recall: {recall:.1f}%)
- **Defender False Positives**: {total_fp} (Precision: {precision:.1f}%)
- **Final Model Integrity**: {env.global_accuracy:.1%}
"""

    log_end(env.global_accuracy)
    grader_results = grader_summary(history, num_rounds, difficulty)
    
    result += f"\n\n**Graders:**\n"
    result += f"Task 1: {grader_results['tasks']['Task 1 (Easy - Detection Recall)']}\n"
    result += f"Task 2: {grader_results['tasks']['Task 2 (Medium - Pattern Precision)']}\n"
    result += f"Task 3: {grader_results['tasks']['Task 3 (Hard - Adversarial Resilience & Speed)']}\n"

    return result, df, fig

def get_client_status_table():
    data = []
    for i in range(env.num_clients):
        is_poison = i in getattr(env, 'poisoner_ids', [])
        status = "🟥 Poisoned" if is_poison else "🟩 Benign"
        data.append({
            "Client ID": f"C-{i:02d}",
            "Status": status,
            "Current Norm": f"{np.mean([u[i] for u in env.client_updates]) if env.client_updates else 0:.3f}",
            "Note": "Targeted by Attacker" if is_poison else ""
        })
    return pd.DataFrame(data)

def update_poison_client_choices(num_clients):
    num_clients = int(num_clients)
    valid_choices = list(range(num_clients))
    default_values = [i for i in [3, 7, 12] if i < num_clients]
    return gr.update(choices=valid_choices, value=default_values)

# # --- GRADIO UI CONFIGURATION ---
# custom_theme = gr.themes.Soft(
#     primary_hue="blue",
#     secondary_hue="blue",
#     neutral_hue="slate",
#     spacing_size="md",
#     radius_size="lg",
# )

# with gr.Blocks(
#     title="Red-Blue Adversarial Federated Learning",
#     theme=custom_theme,
#     css="""
#     .gradio-container { background-color: #f1f5f9 !important; }
#     .prose h1, .prose h2 { color: #1e40af !important; font-weight: 700; }
#     """
# ) as demo:
#     gr.Markdown("# Red-Blue Adversarial Federated Learning\n"
#                 "**LLM-powered Attacker (Red) vs Defender (Blue)**")

#     with gr.Tabs():
#         with gr.Tab("Environment"):
#             with gr.Row():
#                 poison_clients = gr.CheckboxGroup(
#                     choices=list(range(30)),
#                     value=[3, 7, 12],
#                     label="Initial Poisoned Clients"
#                 )
#                 num_clients_dropdown = gr.Dropdown([10, 20, 30], value=20, label="Number of Clients")
#                 num_clients_dropdown.change(fn=update_poison_client_choices, inputs=num_clients_dropdown, outputs=poison_clients)
#                 num_rounds_dropdown = gr.Dropdown([10, 15, 20], value=15, label="Number of Rounds")
#                 difficulty_dropdown = gr.Dropdown(["Easy", "Medium", "Hard"], value="Medium", label="Difficulty Level")
#                 run_btn = gr.Button("🚀 Run Full Episode", variant="primary", size="large")

#             with gr.Row():
#                 result_text = gr.Textbox(label="Episode Summary", interactive=False, lines=6)
#                 accuracy_plot = gr.Plot(label="Agent Performance Dashboard")

#         with gr.Tab("Client Monitor"):
#             gr.Markdown("### Live Client Status")
#             client_table = gr.DataFrame(label="Client Status Table")

#     run_btn.click(
#         fn=run_full_episode,
#         inputs=[poison_clients, num_clients_dropdown, num_rounds_dropdown, difficulty_dropdown],
#         outputs=[result_text, client_table, accuracy_plot]
#     )

# if __name__ == "__main__":
#     demo.launch()