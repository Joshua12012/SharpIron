# app.py
"""
Federated Poisoning Detector - OpenEnv Style + Nice Gradio UI
Fixed: Dynamic number of clients + Proper config update + Index safety
"""

# import gradio as gr
# import pandas as pd
# import plotly.graph_objects as go
# import numpy as np

# # Absolute imports
# from models import FederatedPoisoningAction
# from federated_RLAgent_environment import FederatedPoisoningEnv

# # Global environment instance
# env = FederatedPoisoningEnv()

# def run_episode(selected_poisoners, num_clients, num_rounds):
#     """Run one full episode with dynamic client count."""
#     # Safely update environment configuration
#     num_clients = int(num_clients)
#     num_rounds = int(num_rounds)
    
#     # Update environment
#     env.num_clients = num_clients
#     env.num_rounds = num_rounds
#     env.set_config(selected_poisoners, num_rounds)
    
#     # Reset with new config
#     observation_obj = env.reset()
#     obs = observation_obj.observation
#     info = observation_obj.metadata
#     history = []
#     acc_curve = [env.current_acc]
    
#     for r in range(env.num_rounds):
#         # Generate action safely within current num_clients
#         action = np.random.randint(0, 3, size=env.num_clients)
        
#         # Step the environment
#         step_out = env.step(FederatedPoisoningAction(action_list=action))
#         obs = step_out.observation
#         reward = step_out.reward
#         done = step_out.done
#         info = step_out.metadata
        
#         acc_curve.append(info['accuracy'])
        
#         history.append({
#             "Round": r + 1,
#             "Actions": " | ".join(["Keep" if a==0 else "Flag" if a==1 else "Quarantine" for a in action]),
#             "Accuracy": f"{info['accuracy']:.3f}",
#             "Correct": info['correct_dets'],
#             "False Pos": info['false_pos']
#         })
        
#         if done:
#             break

#     df = pd.DataFrame(history)
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(y=acc_curve, mode='lines+markers', name='Accuracy', line=dict(color='#1D9E75')))
#     fig.update_layout(title="Model Accuracy During Episode", xaxis_title="Round", yaxis_title="Accuracy", height=340)
    
#     result = f"Episode finished. Final Accuracy: {env.current_acc:.3f} | Detected {len(env.detected)}/{len(env.poisoner_ids)} poisoners"
    
#     return result, df, fig


# def get_client_status_table():
#     """Create nice formatted client status table."""
#     data = []
#     for i in range(env.num_clients):
#         is_poison = i in env.poisoner_ids
#         suspicion = env.suspicion[i] if hasattr(env, 'suspicion') else 0.0
        
#         status = "Poisoned" if is_poison else "Benign"
#         color_class = "red" if is_poison else "green"
        
#         data.append({
#             "Client ID": f"C-{i:02d}",
#             "Status": status,
#             "Suspicion": f"{suspicion:.2f}",
#             "Anomaly": f"{env.client_stats[i,4]:.2f}" if hasattr(env, 'client_stats') else "N/A",
#             "Norm": f"{env.client_stats[i,0]:.2f}" if hasattr(env, 'client_stats') else "N/A",
#             "Similarity": f"{env.client_stats[i,1]:.2f}" if hasattr(env, 'client_stats') else "N/A"
#         })
    
#     df = pd.DataFrame(data)
#     return df

# def update_poison_client_choices(num_clients):
#     num_clients = int(num_clients)

#     valid_choices = list(range(num_clients))

#     default_values = [i for i in [2, 5, 7] if i < num_clients]

#     return gr.update(
#         choices=valid_choices,
#         value=default_values
#     )

# with gr.Blocks(title="Federated Poisoning Detector", theme=gr.themes.Soft()) as demo:
#     gr.Markdown("# Federated Poisoning Detector\n"
#                 "RL Agent learning to detect adaptive Byzantine attackers in Federated Learning")
    
#     with gr.Tabs():
#         with gr.Tab("Environment"):
#             with gr.Row():
#                 poison_clients = gr.CheckboxGroup(
#                     choices=list(range(10)),
#                     value=[2, 5, 7],
#                     label="Select Poisoned Clients"
#                 )
#                 num_clients_dropdown = gr.Dropdown(
#                     choices=[10, 20, 30],
#                     value=10,
#                     label="Number of Clients"
#                 )
#                 num_clients_dropdown.change(
#                     fn=update_poison_client_choices,
#                     inputs=num_clients_dropdown,
#                     outputs=poison_clients
#                 )
#                 num_rounds_dropdown = gr.Dropdown(
#                     choices=[10, 50, 100],
#                     value=10,
#                     label="Number of Rounds"
#                 )
#                 run_btn = gr.Button("Run Episode", variant="primary", size="large")
            
#             with gr.Row():
#                 result_text = gr.Textbox(label="Episode Result", interactive=False)
#                 accuracy_plot = gr.Plot(label="Accuracy Curve")
        
#         with gr.Tab("Client Monitor"):
#             gr.Markdown("### Live Client Status")
#             client_table = gr.DataFrame(label="Client Status Table")
        
#         with gr.Tab("Metrics"):
#             metrics_text = gr.Textbox(label="Summary", interactive=False)
        
#         with gr.Tab("About"):
#             gr.Markdown("OpenEnv environment for Byzantine attack detection.\n"
#                         "Agent observes statistical spikes and learns to detect adaptive poisoners.")

#     # Button action
#     run_btn.click(
#         fn=run_episode,
#         inputs=[poison_clients, num_clients_dropdown, num_rounds_dropdown],
#         outputs=[result_text, client_table, accuracy_plot]
#     )

#     # Optional: Update client table when config changes
#     def refresh_client_table():
#         return get_client_status_table()

#     # You can call refresh_client_table() after run if needed

# if __name__ == "__main__":
#     demo.launch(share=True)


# app.py
"""
Federated Poisoning Detector - OpenEnv Style + Nice Gradio UI
Updated with dynamic clients/rounds and better table rendering
"""

# app.py
"""
Federated Poisoning Detector - OpenEnv Style + Nice Gradio UI
Now with actual PPO training (not just fake delay)
"""

# app.py
"""
Federated Poisoning Detector - OpenEnv + Gradio UI + PPO Training
Fixed: Gymnasium wrapper for stable-baselines3
"""

# app.py
"""
Federated Poisoning Detector - OpenEnv Style + Gradio UI + PPO Training
Fixed: Gradio event handler mismatch + Gymnasium wrapper for PPO
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random, os
import gymnasium as gym
from gymnasium import spaces

# PPO imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Your project imports
from models import FederatedPoisoningAction
from federated_RLAgent_environment import FederatedPoisoningEnv


import os
global MODEL_DIR
MODEL_DIR = r"saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
# Global instances
raw_env = FederatedPoisoningEnv()
trained_models = {}




# ====================== GYMNASIUM WRAPPER FOR PPO ======================
# class GymWrapper(gym.Env):
#     """Converts OpenEnv environment into Gymnasium format for stable-baselines3"""
    
#     def __init__(self):
#         super().__init__()
#         self.env = FederatedPoisoningEnv()
        
#         self.observation_space = spaces.Box(
#             low=-10, high=10, 
#             shape=(self.env.num_clients, 7), 
#             dtype=np.float32
#         )
#         self.action_space = spaces.MultiDiscrete([3] * self.env.num_clients)
class GymWrapper(gym.Env):
    def __init__(self, num_clients=10):
        super().__init__()

        self.env = FederatedPoisoningEnv()
        self.env.num_clients = num_clients

        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(num_clients, 7),
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([3] * num_clients)

    # def reset(self, seed=None, options=None):
    #     observation_obj = self.env.reset()

    #     obs = observation_obj.observation
    #     info = observation_obj.metadata if hasattr(observation_obj, "metadata") else {}

    #     return np.array(obs, dtype=np.float32), info
    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Random poisoner count between 10% and 40% of clients
        min_poisoners = max(1, int(self.env.num_clients * 0.1))
        max_poisoners = max(min_poisoners, int(self.env.num_clients * 0.4))

        poisoner_count = random.randint(min_poisoners, max_poisoners)

        poisoner_ids = random.sample(
            range(self.env.num_clients),
            poisoner_count
        )

        # Randomize number of rounds too if desired
        self.env.num_rounds = random.choice([10, 20, 30, 50])

        self.env.set_config(poisoner_ids, self.env.num_rounds)

        observation_obj = self.env.reset()

        obs = observation_obj.observation
        info = observation_obj.metadata if hasattr(observation_obj, "metadata") else {}

        return np.array(obs, dtype=np.float32), info
    def step(self, action):
        step_out = self.env.step(FederatedPoisoningAction(action_list=action.tolist()))
        obs = np.array(step_out.observation, dtype=np.float32)
        reward = step_out.reward
        done = step_out.done
        info = step_out.metadata
        return obs, reward, done, False, info


# ====================== TRAINING FUNCTION ======================
def train_ppo_agent(timesteps: int, num_clients: int):
    global trained_models

    num_clients = int(num_clients)

    def make_env():
        return GymWrapper(num_clients=num_clients)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        device="cpu"
    )

    model.learn(total_timesteps=timesteps)

    trained_models[num_clients] = model

    model_path = os.path.join(MODEL_DIR, f"ppo_model_{num_clients}_clients.zip")
    model.save(model_path)
    trained_models[num_clients] = model
    
    return f"✅ PPO model trained for {num_clients} clients"

def load_saved_models():
    global trained_models

    for num_clients in [10, 20, 30]:
        model_path = os.path.join(MODEL_DIR, f"ppo_model_{num_clients}_clients.zip")

        if os.path.exists(model_path):
            try:
                env = DummyVecEnv([
                    lambda n=num_clients: GymWrapper(num_clients=n)
                ])
                model = PPO.load(model_path, env=env)
                trained_models[num_clients] = model
                print(f"Loaded PPO model for {num_clients} clients")
            except Exception as e:
                print(f"Could not load model for {num_clients}: {e}")

# ====================== RUN EPISODE ======================
# def run_episode(selected_poisoners, num_clients, num_rounds, use_trained):
#     """Run one full episode"""
#     global trained_models
    
#     num_clients = int(num_clients)
#     num_rounds = int(num_rounds)
    
#     raw_env.num_clients = num_clients
#     raw_env.num_rounds = num_rounds
#     raw_env.set_config(selected_poisoners, num_rounds)
    
#     observation_obj = raw_env.reset()
#     obs = observation_obj.observation
#     info = observation_obj.metadata
    
#     history = []
#     acc_curve = [raw_env.current_acc]
    
#     for r in range(raw_env.num_rounds):
#         if use_trained:
#             model = trained_models.get(num_clients)

#             if model is None:
#                 return (
#                     f"❌ No trained PPO model available for {num_clients} clients. Train one first.",
#                     pd.DataFrame(),
#                     go.Figure()
#                 )

#             action, _ = model.predict(
#                 np.array(obs, dtype=np.float32),
#                 deterministic=True
#             )
#         else:
#             action = np.random.randint(0, 3, size=num_clients)
        
#         step_out = raw_env.step(FederatedPoisoningAction(action_list=action.tolist()))
        
#         obs = step_out.observation
#         reward = step_out.reward
#         done = step_out.done
#         info = step_out.metadata
        
#         acc_curve.append(info['accuracy'])
        
#         history.append({
#             "Round": r + 1,
#             "Actions": " | ".join(["Keep" if a==0 else "Flag" if a==1 else "Quarantine" for a in action]),
#             "Accuracy": f"{info['accuracy']:.3f}",
#             "Correct": info['correct_dets'],
#             "False Pos": info['false_pos']
#         })
        
#         if done:
#             break

#     df = pd.DataFrame(history)
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(y=acc_curve, mode='lines+markers', name='Accuracy', line=dict(color='#1D9E75')))
#     fig.update_layout(title="Model Accuracy During Episode", xaxis_title="Round", yaxis_title="Accuracy", height=340)
    
#     status = "🧠 Trained Agent" if (use_trained and trained_models is not None) else "🎲 Random Agent"
#     result = f"{status} | Final Accuracy: {raw_env.current_acc:.3f} | Detected {len(raw_env.detected)}/{len(raw_env.poisoner_ids)} poisoners"
    
#     return result, df, fig

def run_episode(selected_poisoners, num_clients, num_rounds, use_trained):
    """Run one full episode - reward logic moved to env.py"""
    global trained_models
    
    num_clients = int(num_clients)
    num_rounds = int(num_rounds)
    
    raw_env.num_clients = num_clients
    raw_env.num_rounds = num_rounds
    raw_env.set_config(selected_poisoners, num_rounds)
    
    observation_obj = raw_env.reset()
    obs = observation_obj.observation
    info = observation_obj.metadata
    
    history = []
    acc_curve = [raw_env.current_acc]
    
    for r in range(raw_env.num_rounds):
        if use_trained:
            model = trained_models.get(num_clients)

            if model is None:
                return (
                    f"❌ No trained PPO model available for {num_clients} clients. Train one first.",
                    pd.DataFrame(),
                    go.Figure()
                )

            action, _ = model.predict(
                np.array(obs, dtype=np.float32),
                deterministic=True
            )
        else:
            action = np.random.randint(0, 3, size=num_clients)
        
        step_out = raw_env.step(FederatedPoisoningAction(action_list=action.tolist()))
        
        obs = step_out.observation
        reward = step_out.reward
        done = step_out.done
        info = step_out.metadata
        
        # Only update accuracy for UI display (reward logic is now in env)
        acc_curve.append(info['accuracy'])
        
        history.append({
            "Round": r + 1,
            "Actions": " | ".join(["Keep" if a==0 else "Flag" if a==1 else "Quarantine" for a in action]),
            "Accuracy": f"{info['accuracy']:.3f}",
            "Correct": info.get('correct_dets', 0),
            "False Pos": info.get('false_pos', 0)
        })
        
        if done:
            break

    df = pd.DataFrame(history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=acc_curve, mode='lines+markers', name='Accuracy', line=dict(color='#1D9E75')))
    fig.update_layout(title="Model Accuracy During Episode", xaxis_title="Round", yaxis_title="Accuracy", height=340)
    
    status = "🧠 Trained Agent" if (use_trained and trained_models is not None) else "🎲 Random Agent"
    result = f"{status} | Final Accuracy: {raw_env.current_acc:.3f} | Detected {len(raw_env.detected)} poisoners"
    
    return result, df, fig

def update_poison_client_choices(num_clients):
    num_clients = int(num_clients)

    valid_choices = list(range(num_clients))

    default_values = [i for i in [2, 5, 7] if i < num_clients]

    return gr.update(
        choices=valid_choices,
        value=default_values
    )

# ====================== GRADIO UI ======================
with gr.Blocks(title="Federated Poisoning Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Federated Poisoning Detector\n"
                "RL Agent learning to detect adaptive Byzantine attackers in Federated Learning")
    
    with gr.Tabs():
        with gr.Tab("Environment"):
            with gr.Row():
                poison_clients = gr.CheckboxGroup(
                    choices=list(range(30)),
                    value=[2, 5, 7],
                    label="Select Poisoned Clients"
                )
                num_clients_dropdown = gr.Dropdown(
                    choices=[10, 20, 30],
                    value=10,
                    label="Number of Clients"
                )
                num_clients_dropdown.change(
                    fn=update_poison_client_choices,
                    inputs=num_clients_dropdown,
                    outputs=poison_clients
                )
                num_rounds_dropdown = gr.Dropdown(
                    choices=[10, 50, 100],
                    value=10,
                    label="Number of Rounds"
                )
                use_trained = gr.Checkbox(label="Use Trained PPO Agent", value=False)
                run_btn = gr.Button("Run Episode", variant="primary", size="large")
            
            with gr.Row():
                result_text = gr.Textbox(label="Episode Result", interactive=False)
                accuracy_plot = gr.Plot(label="Accuracy Curve")
        
        with gr.Tab("Train Agent"):
            with gr.Row():
                timesteps_slider = gr.Slider(
                    2000, 30000,
                    value=8000,
                    step=500,
                    label="Training Timesteps"
                )

                train_num_clients = gr.Dropdown(
                    choices=[10, 20, 30],
                    value=10,
                    label="Train PPO For Number of Clients"
                )

                train_btn = gr.Button(
                    "🚀 Start PPO Training",
                    variant="primary"
                )

                train_status = gr.Textbox(
                    label="Training Status",
                    interactive=False
                )

                train_btn.click(
                    fn=train_ppo_agent,
                    inputs=[timesteps_slider, train_num_clients],
                    outputs=[train_status]
                )
                train_poison_clients = gr.CheckboxGroup(
                    choices=list(range(30)),
                    value=[2, 5, 7],
                    label="Training Poisoned Clients"
                )
        
        with gr.Tab("Client Monitor"):
            gr.Markdown("### Live Client Status")
            client_table = gr.DataFrame(label="Client Status Table")

    # Button actions - IMPORTANT: 4 inputs now
    run_btn.click(
        fn=run_episode,
        inputs=[poison_clients, num_clients_dropdown, num_rounds_dropdown, use_trained],
        outputs=[result_text, client_table, accuracy_plot]
    )
    
    train_btn.click(
        fn=train_ppo_agent,
        inputs=[timesteps_slider, train_num_clients],
        outputs=[train_status]
    )

if __name__ == "__main__":
    load_saved_models()
    demo.queue()
    demo.launch(
        share=True,
        server_name="localhost",
        server_port=8000,
        show_error=True
    )

