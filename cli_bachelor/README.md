# Connect4 RL — Unified Gymnasium Edition

All training scripts and evaluation now use **`Connect4Env`** (a standard
Gymnasium environment) as the single game interface. Direct `Connect4Game`
calls have been removed from every training loop.

---

## Project structure

```
.
├── game/
│   ├── connect4.py          # Core game logic (unchanged)
│   └── connect4_env.py      # Gymnasium wrapper — single source of truth
├── agents/
│   ├── agents_interface.py
│   ├── reinforcement_agent.py   # DQNAgent + Connect4Net / Connect4NetLegacy
│   ├── rule_based_agent.py
│   ├── minmax_agent.py
│   ├── alfabetapruning_agent.py
│   ├── mcts_agent.py
│   ├── random_agent.py
│   ├── llm_agent.py
│   └── evaluation.py
├── train_dqn/
│   ├── train_dqn.py               # DQN vs RuleBasedAgent  (Gymnasium)
│   ├── train_dqn_switching.py     # Switching self-play     (Gymnasium)
│   └── train_dqn_pure_self.py     # Pure self-play          (Gymnasium)
├── checkpoints/                   # Saved model weights
├── play_gym.py                    # Interactive agent-vs-agent demo
├── tournament.py
├── quick_start_examples.py
└── rl_training_workflow.py
```

---

## Gymnasium API recap

```python
from game.connect4_env import Connect4Env

env = Connect4Env()                         # no opponent — manual 2-player
env = Connect4Env(opponent_agent=some_agent)  # env handles opponent moves

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)  # action = column 0-6
env.close()
```

**Observation** `Box(0,2, shape=(6,7), dtype=int8)` — 0=empty, 1=P1, 2=P2  
**Action** `Discrete(7)` — column index  
**Reward** `+1` win · `-1` loss (with opponent set) · `0` draw · `-10` illegal move

---

## Training scripts

### vs RuleBasedAgent
```bash
python train_dqn/train_dqn.py
python train_dqn/train_dqn.py --episodes 60000 --lr 1e-4
```
Saves: `checkpoints/best_model_rule_based_gym.pt`

### Switching self-play (per-player trajectory buffers, PER, NoisyNet)
```bash
python train_dqn/train_dqn_switching.py
python train_dqn/train_dqn_switching.py --episodes 30000 --use_noisy
```
Saves: `checkpoints/best_model_switching_gym.pt`

### Pure self-play (single-player perspective, PER, NoisyNet)
```bash
python train_dqn/train_dqn_pure_self.py
python train_dqn/train_dqn_pure_self.py --resume checkpoints/best_model_switching_gym.pt
```
Saves: `checkpoints/best_model_pureself_gym.pt`

---

## Playing / demos

```bash
python play_gym.py          # choose agents interactively
python tournament.py        # run a round-robin tournament
python quick_start_examples.py
```

---

## What changed from the original

| File | Change |
|---|---|
| `train_dqn/train_dqn.py` | Replaced `Connect4Game` loop with `Connect4Env`; opponent handled via `env.step()` |
| `train_dqn/train_dqn_switching.py` | Replaced `Connect4Game` with `Connect4Env`; eval also uses env |
| `train_dqn/train_dqn_pure_self.py` | Fixed broken import (`from connect4_env` → `from game.connect4_env`) |
| Everything else | Unchanged |
