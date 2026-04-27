from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.mcts_agent import MCTSAgent
from agents.minmax_agent import MinMaxAgent
from agents.alfabetapruning_agent import AlphaBetaAgent
from agents.llm_agent import LLMAgent
import torch

AGENT_REGISTRY = {
    "random": {
        "label": "Random Agent",
        "description": "Picks a random valid column.",
        "params": {}
    },
    "rule_based": {
        "label": "Rule-Based Agent",
        "description": "Uses handcrafted heuristics: threats, traps, parity.",
        "params": {}
    },
    "mcts": {
        "label": "MCTS Agent",
        "description": "Monte Carlo Tree Search with guided rollouts.",
        "params": {
            "max_iterations": {
                "type": "int", "default": 1000, "min": 100, "max": 20000,
                "label": "Max Iterations"
            }
        }
    },
    "minmax": {
        "label": "MinMax Agent",
        "description": "Minimax search with transposition table.",
        "params": {
            "max_depth": {
                "type": "int", "default": 5, "min": 1, "max": 10,
                "label": "Search Depth"
            }
        }
    },
    "alphabeta": {
        "label": "Alpha-Beta Agent",
        "description": "Minimax with alpha-beta pruning for faster search.",
        "params": {
            "max_depth": {
                "type": "int", "default": 6, "min": 1, "max": 12,
                "label": "Search Depth"
            }
        }
    },
    "llm": {
        "label": "LLM Agent",
        "description": "Local LLM-based agent using Ollama (requires Ollama running locally).",
        "params": {
            "model": {
                "type": "string", "default": "gemma3",
                "label": "Ollama Model Name"
            }
        }
    },
}

# DQN is optional (requires torch)
try:
    from agents.reinforcement_agent import DQNAgent
    import os as _os
    _DEFAULT_MODEL_PATH = _os.path.join(_os.path.dirname(__file__), "agents", "best_model.pt")
    AGENT_REGISTRY["dqn"] = {
        "label": "DQN Agent",
        "description": "Deep Q-Network reinforcement learning agent (trained CNN policy).",
        "params": {
            "model_path": {
                "type": "string", "default": _DEFAULT_MODEL_PATH,
                "label": "Model Path (optional)"
            },
            # "epsilon": {
            #     "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
            #     "label": "Epsilon (exploration)"
            # }
        }
    }
except ImportError:
    pass


def create_agent(agent_type: str, player_id: int, config: dict = None):
    config = config or {}

    if agent_type == "random":
        return RandomAgent(player_id)
    elif agent_type == "rule_based":
        return RuleBasedAgent(player_id)
    elif agent_type == "mcts":
        return MCTSAgent(
            player_id,
            max_iterations=int(config.get("max_iterations", 1000))
        )
    elif agent_type == "minmax":
        return MinMaxAgent(
            player_id,
            max_depth=int(config.get("max_depth", 5))
        )
    elif agent_type == "alphabeta":
        return AlphaBetaAgent(
            player_id,
            max_depth=int(config.get("max_depth", 6))
        )
    elif agent_type == "dqn":
        from agents.reinforcement_agent import DQNAgent
        return DQNAgent(
            player_id,
            model_path=config.get("model_path") or None,
            # epsilon=float(config.get("epsilon", 0.0))
        )
    elif agent_type == "llm":
        return LLMAgent(
            player_id,
            model=config.get("model", "gemma3")
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")