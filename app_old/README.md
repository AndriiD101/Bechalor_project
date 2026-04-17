# Connect 4 Arena

A full-stack Connect 4 platform with Human vs Human, Human vs Agent, and Agent vs Agent modes.

## Features

- **3 game modes**: Human vs Human, Human vs Agent, Agent vs Agent
- **5 AI agents**: Random, Rule-Based, MCTS, MinMax, Alpha-Beta (+ optional DQN)
- **Configurable agent parameters**: depth, iterations, epsilon, etc.
- **Persistent game history**: all outcomes saved to SQLite
- **Stats dashboard**: win rates, mode distribution
- **Polished UI**: retro-arcade aesthetic, animated board, move log, replay viewer

## Quick Start (Docker)

```bash
docker-compose up --build
```

Open **http://localhost:8000** in your browser.

## Local Development (without Docker)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000**

## Project Structure

```
connect4/
├── backend/
│   ├── main.py               # FastAPI app
│   ├── database.py           # SQLAlchemy models + DB setup
│   ├── session_manager.py    # In-memory active game sessions
│   ├── agent_factory.py      # Agent registry + factory
│   ├── requirements.txt
│   ├── game/
│   │   └── connect4.py       # Game logic
│   ├── agents/
│   │   ├── agents_interface.py
│   │   ├── random_agent.py
│   │   ├── rule_based_agent.py
│   │   ├── mcts_agent.py
│   │   ├── minmax_agent.py
│   │   ├── alfabetapruning_agent.py
│   │   ├── reinforcement_agent.py  # DQN (requires torch)
│   │   └── evaluation.py
│   └── routers/
│       └── game.py           # All API endpoints
├── frontend/
│   └── index.html            # Single-page UI
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agents | List available agents + params |
| POST | /api/games | Create new game |
| GET | /api/games/{id} | Get game state |
| POST | /api/games/{id}/move | Human move |
| POST | /api/games/{id}/agent-move | Step agent (AvA) |
| POST | /api/games/{id}/auto-play | Run full AvA game |
| DELETE | /api/games/{id} | Abandon game |
| GET | /api/history | Recent game records |
| GET | /api/history/{id} | Game detail with board |
| GET | /api/stats | Aggregate statistics |

## Agent Parameters

| Agent | Parameters |
|-------|-----------|
| Random | None |
| Rule-Based | None |
| MCTS | `max_iterations` (100–20000, default 1000) |
| MinMax | `max_depth` (1–10, default 5) |
| Alpha-Beta | `max_depth` (1–12, default 6) |
| DQN | `model_path` (optional), `epsilon` (0.0–1.0) |

## DQN Agent

The DQN agent requires PyTorch. Add `torch` to `requirements.txt` and provide a trained model checkpoint path in the agent configuration field.

## Notes

- All games are automatically saved to the database when finished
- Active games are held in memory; only completed games are persisted
- The AvA mode supports step-by-step and auto-play with speed control
