# app.py
"""
Red-Blue Adversarial Federated Learning UI
With strict [START]/[STEP]/[END] logging for evaluation + LLM prompt/response visibility
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

# For debugging LLM calls
DEBUG_LLM = True




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
    """Main episode runner with strict logging"""
    log_start()

    num_clients = int(num_clients)
    num_rounds = int(num_rounds)
    difficulty = difficulty.lower()

    env.num_clients = num_clients
    env.num_rounds = num_rounds
    env.set_config(selected_poisoners, num_rounds, difficulty=difficulty)

    obs = env.reset()
    history = []
    acc_curve = [env.global_accuracy]

    for r in range(num_rounds):
        # Create a light version of the observation for the LLMs (No raw floats)
        light_obs = obs.dict() if hasattr(obs, 'dict') else obs.copy()
        if "client_updates" in light_obs:
            del light_obs["client_updates"]

        # Attacker acts
        attacker_action = attacker_agent.act(light_obs, difficulty)

        # Defender acts
        defender_action = defender_agent.act(light_obs)

        # Environment step
        observation, attacker_reward, defender_reward, info = env.step(attacker_action, defender_action)
        
        # Give feedback to both agents
        attacker_agent.update_feedback(attacker_reward, info.get("attacker_reason", ""))
        defender_agent.update_feedback(defender_reward, info.get("defender_reason", ""), defender_action.target_clients)
        
        log_step(r+1, attacker_action, defender_action, observation, attacker_reward, defender_reward)

        acc_curve.append(observation.global_accuracy)

        # FIX: Align keys and descriptions with graders.py requirements
        history.append({
            "Round": r + 1,
            "Attacker Action": f"{attacker_action.attack_type} on clients {attacker_action.target_clients}",
            "Defender Action": f"{defender_action.action_type} on clients {defender_action.target_clients}",
            "Accuracy": f"{observation.global_accuracy:.3f}",
            "A_Reward": f"{attacker_reward:.2f}",
            "D_Reward": f"{defender_reward:.2f}"
        })

        obs = observation
        time.sleep(0.35)

    df = pd.DataFrame(history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=acc_curve, mode='lines+markers', name='Accuracy', line=dict(color='#1D9E75')))
    fig.update_layout(title=f"Global Accuracy - {difficulty.capitalize()} Difficulty", 
                      xaxis_title="Round", yaxis_title="Accuracy", height=380)

    result = f"""
**Episode Complete**  
Difficulty: **{difficulty.capitalize()}**  
Final Accuracy: **{env.global_accuracy:.3f}**
"""

    log_end(env.global_accuracy)
        # Compute graders
    grader_results = grader_summary(history, env.global_accuracy, num_rounds)
    
    result += f"\n\n**Graders:**\n"
    result += f"Task 1 (Easy): {grader_results['tasks']['Task 1 (Easy - Accuracy Preservation)']}\n"
    result += f"Task 2 (Medium): {grader_results['tasks']['Task 2 (Medium - Pattern Recognition)']}\n"
    result += f"Task 3 (Hard): {grader_results['tasks']['Task 3 (Hard - Robust Defense)']}\n"

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


# Replace this part in your app.py (near the bottom, where the UI starts)

# At the beginning of the UI section in app.py

# After all imports, before the functions, add this:

# custom_theme = gr.themes.Soft(
#     primary_hue="blue",
#     secondary_hue="blue",
#     neutral_hue="slate"
# )

# with gr.Blocks(
#     title="Red-Blue Adversarial Federated Learning",
#     theme=custom_theme,
#     css="""
#     .gradio-container { background-color: #f8fafc !important; }
#     .prose h1 { color: #1e40af !important; font-weight: 600; }
#     """
# ) as demo:
#     gr.Markdown("# Red-Blue Adversarial Federated Learning\n"
#                 "**LLM-powered Attacker (Red) vs Defender (Blue)**")

# Replace your current Blocks creation with this:

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    spacing_size="md",
    radius_size="lg",
)

with gr.Blocks(
    title="Red-Blue Adversarial Federated Learning",
    theme=custom_theme,
    css="""
    .gradio-container {
        background-color: #f1f5f9 !important;   /* Soft light gray background */
    }
    .prose h1, .prose h2 {
        color: #1e40af !important;
        font-weight: 700;
    }
    .gr-button-primary {
        background-color: #1e40af !important;
        border-color: #1e40af !important;
    }
    .gr-textbox, .gr-plot, .gr-dataframe {
        background-color: white !important;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    """
) as demo:
    gr.Markdown("# Red-Blue Adversarial Federated Learning\n"
                "**LLM-powered Attacker (Red) vs Defender (Blue)**")

    with gr.Tabs():
        with gr.Tab("Environment"):
            with gr.Row():
                poison_clients = gr.CheckboxGroup(
                    choices=list(range(30)),
                    value=[3, 7, 12],
                    label="Initial Poisoned Clients"
                )
                num_clients_dropdown = gr.Dropdown([10, 20, 30], value=20, label="Number of Clients")
                num_clients_dropdown.change(fn=update_poison_client_choices, inputs=num_clients_dropdown, outputs=poison_clients)
                num_rounds_dropdown = gr.Dropdown([10, 15, 20], value=15, label="Number of Rounds")
                difficulty_dropdown = gr.Dropdown(["Easy", "Medium", "Hard"], value="Medium", label="Difficulty Level")
                run_btn = gr.Button("🚀 Run Full Episode", variant="primary", size="large")

            with gr.Row():
                result_text = gr.Textbox(label="Episode Summary", interactive=False, lines=6)
                accuracy_plot = gr.Plot(label="Global Accuracy Curve")

        with gr.Tab("Client Monitor"):
            gr.Markdown("### Live Client Status")
            client_table = gr.DataFrame(label="Client Status Table")

    run_btn.click(
        fn=run_full_episode,
        inputs=[poison_clients, num_clients_dropdown, num_rounds_dropdown, difficulty_dropdown],
        outputs=[result_text, client_table, accuracy_plot]
    )

# if __name__ == "__main__":
#     demo.launch(share=True)