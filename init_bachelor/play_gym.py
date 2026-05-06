"""
play_gym.py – demo of Connect4Env through the standard Gymnasium API with agent selection.

Run:
    python play_gym.py

Requirements:
    pip install gymnasium numpy
"""

from game.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.minmax_agent import MinMaxAgent
from agents.alfabetapruning_agent import AlphaBetaAgent
from agents.mcts_agent import MCTSAgent
from agents.reinforcement_agent import DQNAgent


def create_agent(agent_type: str, player_id: int):
    """Create an agent based on the user's choice."""
    if agent_type.lower() == "random":
        return RandomAgent(player_id)
    elif agent_type.lower() == "rule_based":
        return RuleBasedAgent(player_id)
    elif agent_type.lower() == "minmax":
        depth = input(f"  Enter max_depth for MinMax (default 4): ").strip()
        depth = int(depth) if depth.isdigit() else 4
        return MinMaxAgent(player_id, max_depth=depth)
    elif agent_type.lower() == "alphabeta":
        depth = input(f"  Enter max_depth for AlphaBeta (default 4): ").strip()
        depth = int(depth) if depth.isdigit() else 4
        return AlphaBetaAgent(player_id, max_depth=depth)
    elif agent_type.lower() == "mcts":
        iterations = input(f"  Enter max_iterations for MCTS (default 10000): ").strip()
        iterations = int(iterations) if iterations.isdigit() else 10000
        return MCTSAgent(player_id, max_iterations=iterations)
    elif agent_type.lower() == "dqn":
        model_path = input(f"  Enter model path (default: checkpoints/best_model_cur.pt): ").strip()
        if not model_path:
            model_path = "checkpoints/best_model_cur.pt"
        return DQNAgent(player_id, model_path=model_path, epsilon=0.0)
    else:
        print("Unknown agent type, defaulting to RandomAgent")
        return RandomAgent(player_id)


def select_agent(player_num: int) -> str:
    """Display agent options and let user select."""
    print(f"\n--- Select Agent for Player {player_num} ---")
    print("1. Random")
    print("2. Rule-Based")
    print("3. MinMax")
    print("4. AlphaBeta")
    print("5. MCTS")
    print("6. DQN")
    
    choice = input(f"Enter choice (1-6) for Player {player_num}: ").strip()
    
    agents_map = {
        "1": "random",
        "2": "rule_based",
        "3": "minmax",
        "4": "alphabeta",
        "5": "mcts",
        "6": "dqn",
    }
    
    return agents_map.get(choice, "random")


if __name__ == "__main__":
    print("=" * 60)
    print("Connect4 Gymnasium Environment - Agent Selection")
    print("=" * 60)
    
    # Let user choose agents
    agent1_type = select_agent(1)
    print(f"✓ Player 1 selected: {agent1_type.upper()}")
    
    agent2_type = select_agent(2)
    print(f"✓ Player 2 selected: {agent2_type.upper()}")
    
    # Create agents
    print("\nCreating agents...")
    agent1 = create_agent(agent1_type, player_id=1)
    agent2 = create_agent(agent2_type, player_id=2)
    
    print(f"✓ Agent 1: {type(agent1).__name__}")
    print(f"✓ Agent 2: {type(agent2).__name__}")
    
    # Setup environment with Agent 1 as player and Agent 2 as opponent
    print("\nStarting game...\n")
    env = Connect4Env(opponent_agent=agent2, render_mode="human")
    obs, info = env.reset()
    
    terminated = truncated = False
    move_count = 0
    while not (terminated or truncated):
        # Your agent (player 1) picks a column
        action = agent1.select_move(env.game)
        obs, reward, terminated, truncated, info = env.step(action)
        move_count += 1
    
    print("=" * 60)
    print(f"Game Over!")
    print(f"Winner: {info['winner']}")
    print(f"Moves: {move_count}")
    print("=" * 60)
    env.close()
