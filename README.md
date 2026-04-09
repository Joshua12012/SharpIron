---
title: "SharpernerRL: Federated Adversarial Environment"
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
base_path: /web
---

# SharpernerRL: Adversarial Federated Learning Simulation

SharpernerRL is an advanced simulation environment for **Federated Learning Security**. It provides a platform to evaluate Red-Blue adversarial dynamics where a **Red Team (Attacker)** attempts to poison a global model and a **Blue Team (Defender)** attempts to detect anomalies and quarantine malicious clients.

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file (see `.env.example`) with your API keys:
```env
HF_TOKEN=your_token_here
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

### 3. Run the System
Start the backend server:
```bash
python server/app.py
```
In a new terminal, run the simulation:
```bash
python inference.py
```

---

## 🏗️ Project Structure

This project follows the **OpenEnv** standard for federated reinforcement learning environments.

```text
federated_RLAgent/
├── inference.py          # Main execution loop and evaluation runner
├── client.py             # OpenEnv HTTP client
├── sharperner_env.py     # Environment API wrapper
├── models.py             # Pydantic data models (Actions/Observations)
├── graders.py            # Logic for scoring tasks
├── rewards.py            # Reward calculation logic
├── openenv.yaml          # Manifest for OpenEnv discovery
├── pyproject.toml        # Project metadata
│
├── agents/               # Intelligent Agent Logic
│   ├── attacker.py       # LLM-based Red Team agent
│   └── defender.py       # LLM-based Blue Team agent
│
├── server/               # Simulation Engine & API
│   ├── app.py            # FastAPI server entry point
│   ├── environment.py    # Core simulation mechanics
│   └── static/           # Sentinel Dashboard UI (HTML/JS)
│
└── components/           # Logic for tasks and grades (if applicable)
```

---

## 📊 How It Works

1.  **Environment**: Manages a set of FL clients and a global model.
2.  **Attacker**: Selects targets and applies poisoning strategies (e.g., *stealth*, *coordinated*).
3.  **Defender**: Inspects client telemetry and anomalies to *detect* or *quarantine* nodes.
4.  **Scoring**: Performance is graded across three tasks: **Detection Recall**, **Pattern Precision**, and **Adversarial Resilience**.

### Manual UI Interaction
The environment hosts a **Sharperner Dashboard** reachable at `http://localhost:8000/web` when the server is running. 

![Dashboard Overview](Images/dashboard.png)


---

## 🏆 Baseline Performance

| Task | Baseline Score | Success Threshold |
| :--- | :--- | :--- |
| **Detection Recall (Easy)** | **0.92** | 0.80 |
| **Pattern Precision (Medium)** | **0.84** | 0.75 |
| **Adversarial Resilience (Hard)** | **0.68** | 0.60 |

---

## 🐋 Docker Support
To build and run as a container:
```bash
docker build -t sharperner-rl .
docker run -p 8000:8000 sharperner-rl
```
