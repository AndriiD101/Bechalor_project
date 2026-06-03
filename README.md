# Connect 4 Arena — Full Stack AI Platform

A complete Connect 4 gaming platform featuring multiple AI agents, reinforcement learning, and interactive gameplay. Includes both a **web application** (`app/`) and **CLI training/evaluation tools** (`cli_bachelor/`).

---

## 🎮 Project Overview

This project implements Connect 4 with the following components:

### **app/** — Full-Stack Web Application
A web-based Connect 4 arena with:
- **Flask backend** with REST API
- **SQLite database** for persistent game history
- **Interactive frontend** (HTML/CSS/JS)
- 5 AI agent types for gameplay
- **3 game modes**: Human vs Human, Human vs Agent, Agent vs Agent
- **Stats dashboard** with win rates and game analytics

### **cli_bachelor/** — AI Training & Evaluation Suite
Command-line tools for:
- **DQN training** using Gymnasium API (multiple training paradigms)
- **Tournament evaluation** with configurable agent parameters
- **Interactive demos** with agent-vs-agent gameplay
- **Model checkpoints** for trained neural networks
- **Sweep results** from large-scale agent parameter evaluations

---

## 🏗️ Project Structure

```
bachalor_project/
├── app/                              # Full-stack web application
│   ├── App_flask/                   # Python virtual environment
│   ├── frontend/                    # Web UI (HTML/CSS/JS)
│   │   ├── index.html
│   │   ├── main.js
│   │   └── styles.css
│   └── backend/                     # Flask server & game logic
│       ├── main.py                  # Flask app entry point
│       ├── database.py              # SQLAlchemy models & DB setup
│       ├── session_manager.py       # Game session management
│       ├── agent_factory.py         # Agent instantiation
│       ├── gif_generator.py         # Game replay animation
│       ├── init_db.py               # Database initialization
│       ├── requirements.txt         # Flask + dependencies
│       ├── routers/                 # API endpoints (game routes)
│       ├── agents/                  # Agent implementations
│       ├── game/                    # Core game logic
│       ├── data/                    # SQLite database storage
│       ├── uploaded_models/         # User-uploaded DQN models
│       └── saved_gifs/              # Generated game replays
│
└── cli_bachelor/                     # CLI training & evaluation
    ├── game/
    │   ├── connect4.py              # Core game logic
    │   └── connect4_env.py          # Gymnasium environment wrapper
    ├── agents/                      # AI agent implementations
    │   ├── agents_interface.py      # Base agent interface
    │   ├── random_agent.py
    │   ├── rule_based_agent.py
    │   ├── minmax_agent.py
    │   ├── alfabetapruning_agent.py
    │   ├── mcts_agent.py
    │   ├── llm_agent.py
    │   ├── reinforcement_agent.py   # DQN agent (torch)
    │   └── evaluation.py            # Agent evaluation utilities
    ├── train_dqn/                   # DQN training scripts
    │   ├── train_dqn.py             # DQN vs RuleBasedAgent
    │   ├── train_dqn_switching.py   # Switching self-play mode
    │   └── train_dqn_pure_self.py   # Pure self-play mode
    ├── checkpoints/                 # Saved model weights (.pt files)
    ├── sweep_results/               # Tournament results & agent benchmarks
    ├── info/                        # Reference docs
    │   ├── rules.txt
    │   ├── ref.txt
    │   └── train_time.txt
    ├── python_venv/                 # Python virtual environment
    ├── play_gym.py                  # Interactive agent-vs-agent demo
    ├── tournament.py                # Round-robin tournament script
    ├── quick_start_examples.py      # Getting started examples
    ├── rl_training_workflow.py      # Workflow coordination
    └── requirements.txt             # CLI dependencies (gymnasium, torch, etc.)
```

---

## 🤖 AI Agents

The project includes **6 agent types** with varying complexity:

| Agent | Type | Parameters | Use Case |
|-------|------|-----------|----------|
| **Random** | Baseline | None | Random move selection |
| **Rule-Based** | Heuristic | None | Hand-crafted winning strategies |
| **MinMax** | Tree Search | `max_depth` (1-6) | Exhaustive move evaluation |
| **Alpha-Beta** | Tree Search | `max_depth` (1-6) | Optimized MinMax with pruning |
| **MCTS** | Monte Carlo Tree Search | `max_iterations` (5K-20K) | Probabilistic tree exploration |
| **DQN** | Reinforcement Learning | `model_path` | Neural network trained with deep Q-learning |

---

## 🚀 Getting Started

### **Option 1: Web Application (app/)**

#### Install & Run

```bash
cd app/backend
pip install -r requirements.txt
python main.py
```

Then open `http://localhost:5000` in your browser.

#### Features
- Play against AI agents with configurable difficulty
- View game history and statistics
- Generate replay GIFs of past games
- Multiple game modes (Human vs Human, Human vs Agent, Agent vs Agent)

#### Database Setup
- SQLite database automatically created at `backend/data/connect4.db`
- Schema defined in `backend/schema.sql`
- To reset: `python init_db.py`

---

### **Option 2: CLI Tools (cli_bachelor/)**

#### Install & Run

```bash
cd cli_bachelor
pip install -r requirements.txt
```

#### Available Scripts

**Interactive Demo — Agent vs Agent**
```bash
python play_gym.py
```
Interactively select agents, view parameters, and play a game in the terminal.

**Tournament Evaluation**
```bash
python tournament.py
```
Runs a configurable round-robin tournament comparing all agent variants.
- Customize agents and parameters in `PARAM_SWEEP`
- Results saved to `sweep_results/`

**Quick Start Examples**
```bash
python quick_start_examples.py
```
Runs predefined agent matchups to get you started.

---

## 📚 DQN Training (cli_bachelor/train_dqn/)

Three training paradigms using **Gymnasium API**:

### 1. **DQN vs Rule-Based Agent**
Train DQN against fixed opponent:
```bash
python train_dqn/train_dqn.py --episodes 50000 --lr 1e-4
```
- Saves: `checkpoints/best_model_rule_based_gym.pt`

### 2. **Switching Self-Play**
Agent switches between P1/P2 roles with trajectory buffers:
```bash
python train_dqn/train_dqn_switching.py --episodes 30000 --use_noisy
```
- Uses Prioritized Experience Replay (PER)
- Optional NoisyNet layers for exploration
- Saves: `checkpoints/best_model_switching_gym.pt`

### 3. **Pure Self-Play**
Single-perspective self-play training:
```bash
python train_dqn/train_dqn_pure_self.py --episodes 60000
python train_dqn/train_dqn_pure_self.py --resume checkpoints/best_model_switching_gym.pt
```
- Supports resuming from checkpoint
- Saves: `checkpoints/best_model_pureself_gym.pt`

---

## 🏆 Tournament & Evaluation

### Run Tournaments
```bash
python cli_bachelor/tournament.py
```

**Configuration** (in `tournament.py`):
```python
PARAM_SWEEP = {
    "MinMax": {"max_depth": [1, 2, 3, 4, 5, 6]},
    "AlphaBeta": {"max_depth": [1, 2, 3, 4, 5, 6]},
    "MCTS": {"max_iterations": [5000, 10000, 15000, 20000]},
    "DQN": {"model_path": ["checkpoints/best_model.pt"]},
}
GAMES_PER_MATCHUP = 50  # games per P1-P2 pair
```

Results are saved to `sweep_results/` with format:
```
Agent1_vs_Agent2.txt
```

---

## 🔧 Technologies Used

### **Web Application** (app/)
- **Backend**: Flask, Flask-CORS, SQLAlchemy
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Image Processing**: Pillow (for GIF generation)

### **CLI & Training** (cli_bachelor/)
- **Game API**: Gymnasium (standard RL environment interface)
- **Deep Learning**: PyTorch
- **Tree Search**: NumPy (for MinMax, Alpha-Beta, MCTS algorithms)
- **Training**: Stable-Baselines3 (DQN, PER, NoisyNet support)

---

## 📊 Data & Results

### Checkpoints
Located in `cli_bachelor/checkpoints/`:
- `dqn_single.pt` — DQN trained vs rule-based
- `dqn_self_play.pt` — DQN trained via self-play
- `dqn_curriculum.pt` — DQN with curriculum learning
- `dqn_random_state.pt` — DQN with random initialization

### Sweep Results
Located in `cli_bachelor/sweep_results/`:
- Files compare all agent configurations
- Format: `Agent1[param=value]_vs_Agent2[param=value].txt`
- Contains: game results, win rates, draw counts

---

## 🎓 Bachelor Project Context

This is a **Bachelor thesis project** exploring:
1. **AI Game Playing**: Comparing search-based vs learning-based approaches
2. **Reinforcement Learning**: DQN training in competitive settings
3. **Algorithm Comparison**: MinMax, Alpha-Beta, MCTS vs neural networks
4. **Interactive Systems**: Web UI for human-AI gameplay

Key research areas:
- Training paradigms for self-play RL
- Parameter tuning for tree search algorithms
- Performance benchmarking across agent types

---

## 🛠️ Development

### Adding New Agents
1. Extend `agents_interface.py`
2. Implement `get_move(board)` method
3. Register in `agent_factory.py` (app) and `play_gym.py` (CLI)

### Adding Training Modes
1. Create new script in `cli_bachelor/train_dqn/`
2. Use `Connect4Env` from `game/connect4_env.py`
3. Save checkpoints to `checkpoints/`

### Extending Web Features
1. Add routes to `backend/routers/`
2. Implement frontend logic in `frontend/main.js`
3. Update database schema if needed in `backend/database.py`

---

## 📝 License & Credits

Bachelor project implementation combining game theory, reinforcement learning, and full-stack web development.

---

## 🚀 Quick Commands

```bash
# Web app
cd app/backend && python main.py

# CLI demo
cd cli_bachelor && python play_gym.py

# Tournament
cd cli_bachelor && python tournament.py

# Train DQN
cd cli_bachelor && python train_dqn/train_dqn.py --episodes 50000
```

---

**Happy playing! 🎮**
