"""
Reinforcement Learning Training Workflow for Connect4 using Gymnasium

This script demonstrates multiple approaches to training RL agents on Connect4:
  1. Training with fixed opponent (e.g., Rule-Based agent)
  2. Self-play training (two agents learning against each other)
  3. Vectorized training using AsyncVectorEnv for parallel episodes
  4. Integration with Stable-Baselines3 frameworks

The workflow showcases how the Gymnasium-compliant Connect4Env enables
seamless integration with modern RL libraries.

Usage:
    python rl_training_workflow.py --mode fixed-opponent --opponent rule-based
    python rl_training_workflow.py --mode self-play --episodes 1000
    python rl_training_workflow.py --mode vectorized --num-envs 4
    python rl_training_workflow.py --mode sb3 --timesteps 100000
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from datetime import datetime

# Gymnasium and RL imports
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
try:
    from gymnasium.vector import AsyncVectorEnv
    HAS_VECTOR_ENV = True
except ImportError:
    HAS_VECTOR_ENV = False

# Stable-Baselines3 (optional dependency)
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

import torch
import torch.nn as nn
import torch.optim as optim

from game.connect4_env import Connect4Env
from agents.rule_based_agent import RuleBasedAgent
from agents.random_agent import RandomAgent
from agents.minmax_agent import MinMaxAgent


# ============================================================================ #
#  Logging Setup                                                               #
# ============================================================================ #

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('rl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================ #
#  Simple DQN Implementation                                                   #
# ============================================================================ #

class SimpleQNetwork(nn.Module):
    """
    A simple CNN-based Q-network for Connect4.

    Architecture:
      - 3-channel input (agent pieces, opponent pieces, valid moves mask)
      - 2 convolutional layers
      - Flatten and 2 dense layers
      - Output: 7 Q-values (one per column)
    """

    def __init__(self, input_channels=3, num_actions=7):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Compute flatten size dynamically to avoid hardcoding errors
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 6, 7)
            dummy = torch.relu(self.conv1(dummy))
            dummy = torch.relu(self.conv2(dummy))
            self.flatten_size = dummy.reshape(1, -1).shape[1]
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class DQNTrainer:
    """
    A simple DQN trainer for Connect4 against a fixed opponent.

    This demonstrates basic RL training mechanics: experience replay,
    target network updates, and epsilon-greedy exploration.
    """

    def __init__(self, opponent_agent, learning_rate=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995):
        """
        Args:
            opponent_agent: Agent with select_move(game) method
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate per episode
        """
        self.opponent_agent = opponent_agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Network
        self.network = SimpleQNetwork().to(self.device)
        self.target_network = SimpleQNetwork().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Experience replay buffer
        self.replay_buffer = []
        self.max_buffer_size = 10000

    def board_to_tensor(self, game, from_player=None):
        """
        Convert Connect4Game board to 3-channel tensor from a specific player's perspective.
        
        Args:
            game: Connect4Game instance
            from_player: Which player's perspective (1 or 2). If None, uses current player.
        """
        if from_player is None:
            from_player = game.current_player
            
        board = game.get_board()
        # Channel 0: This agent's pieces
        agent_mask = (board == from_player).astype(np.int8)
        # Channel 1: Opponent's pieces
        opponent_player = 3 - from_player
        opponent_mask = (board == opponent_player).astype(np.int8)
        # Channel 2: Valid move mask
        valid_mask = np.zeros((6, 7), dtype=np.int8)
        for col in game.get_valid_locations():
            valid_mask[5, col] = 1

        tensor = np.stack([agent_mask, opponent_mask, valid_mask], axis=0)
        return torch.from_numpy(tensor).float().unsqueeze(0).to(self.device)

    def select_action(self, game, training=True):
        """Select action using epsilon-greedy policy."""
        valid_cols = game.get_valid_locations()
        if not valid_cols:
            raise ValueError("select_action called on a finished game with no valid moves.")

        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_cols)
        else:
            with torch.no_grad():
                board_tensor = self.board_to_tensor(game)
                q_values = self.network(board_tensor)[0]
                # Mask invalid actions with large negative value
                valid_mask = torch.full((7,), -1e8, device=self.device)
                for col in valid_cols:
                    valid_mask[col] = q_values[col]
                action = valid_mask.argmax().item()
            return action

    def remember(self, state_tensor, action, reward, next_state_tensor, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((state_tensor, action, reward, next_state_tensor, done))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

    def train_on_batch(self, batch_size=32):
        """Train network on random batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute Q-targets
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        # Compute Q-predictions and loss
        q_pred = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_pred, targets)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Sync target network with main network."""
        self.target_network.load_state_dict(self.network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(self):
        """Run one training episode."""
        env = Connect4Env(opponent_agent=self.opponent_agent, render_mode=None)
        obs, info = env.reset()

        state_tensor = self.board_to_tensor(env._game)
        done = False
        total_reward = 0.0

        while not done:
            # Agent acts
            action = self.select_action(env._game, training=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Remember transition
            next_state_tensor = self.board_to_tensor(env._game)
            self.remember(state_tensor, action, reward, next_state_tensor, done)
            state_tensor = next_state_tensor

            # Train on batch
            loss = self.train_on_batch(batch_size=32)

        env.close()
        self.decay_epsilon()
        return total_reward

    def train(self, num_episodes=1000, target_update_freq=100, eval_freq=100):
        """Train the agent."""
        logger.info(f"Starting DQN training for {num_episodes} episodes")
        logger.info(f"Opponent: {self.opponent_agent.__class__.__name__}")

        episode_rewards = []

        for episode in range(num_episodes):
            reward = self.train_episode()
            episode_rewards.append(reward)

            if (episode + 1) % target_update_freq == 0:
                self.update_target_network()

            if (episode + 1) % eval_freq == 0:
                avg_reward = np.mean(episode_rewards[-eval_freq:])
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )

        return episode_rewards

    def save(self, checkpoint_path):
        """Save network weights."""
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

    def load(self, checkpoint_path):
        """Load network weights."""
        self.network.load_state_dict(torch.load(checkpoint_path))
        self.target_network.load_state_dict(self.network.state_dict())
        logger.info(f"Model loaded from {checkpoint_path}")

    def evaluate(self, num_games=100, opponent=None):
        """Evaluate agent against opponent."""
        if opponent is None:
            opponent = self.opponent_agent

        wins = 0
        for _ in range(num_games):
            env = Connect4Env(opponent_agent=opponent, render_mode=None)
            obs, info = env.reset()
            done = False

            while not done:
                action = self.select_action(env._game, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if reward == 1.0:
                    wins += 1

            env.close()

        win_rate = wins / num_games
        logger.info(f"Win rate vs {opponent.__class__.__name__}: {win_rate:.2%} ({wins}/{num_games})")
        return win_rate


# ============================================================================ #
#  Training Workflows                                                          #
# ============================================================================ #

def train_fixed_opponent(opponent_name: str = "rule-based", episodes: int = 500):
    """
    Train an agent against a fixed opponent using simple DQN.

    This approach trains a single agent to beat a specific opponent.
    Good for rapid prototyping and debugging.
    """
    logger.info(f"=== Training against {opponent_name} ===")

    # Create opponent
    if opponent_name == "rule-based":
        opponent = RuleBasedAgent(player_id=2)
    elif opponent_name == "random":
        opponent = RandomAgent(player_id=2)
    elif opponent_name == "minmax":
        opponent = MinMaxAgent(player_id=2, max_depth=3)
    else:
        raise ValueError(f"Unknown opponent: {opponent_name}")

    # Train
    trainer = DQNTrainer(opponent_agent=opponent)
    rewards = trainer.train(num_episodes=episodes, target_update_freq=50, eval_freq=50)

    # Save
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    trainer.save(checkpoint_dir / f"dqn_vs_{opponent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

    # Evaluate
    trainer.evaluate(num_games=100, opponent=opponent)

    logger.info(f"Training complete!")


def train_vectorized_envs(num_envs: int = 4, episodes: int = 500):
    """
    Train using vectorized environments for parallel experience collection.

    This approach collects experience from multiple environments simultaneously,
    improving sample efficiency and training speed.

    Requires gymnasium.vector.AsyncVectorEnv.
    """
    if not HAS_VECTOR_ENV:
        logger.error("Vectorized training requires gymnasium.vector")
        return

    logger.info(f"=== Training with {num_envs} parallel environments ===")

    # Create opponent
    opponent = RuleBasedAgent(player_id=2)

    # Create vectorized environment
    def make_env(opponent_agent):
        def env_fn():
            return Connect4Env(opponent_agent=opponent_agent)
        return env_fn

    env_fns = [make_env(opponent) for _ in range(num_envs)]
    vec_env = AsyncVectorEnv(env_fns)

    logger.info(f"Running {episodes} episodes with {num_envs} parallel environments")

    num_steps = episodes * 50  # Rough estimate of average steps per episode
    obs, info = vec_env.reset()

    for step in range(num_steps):
        # Random actions for demo
        actions = np.array([vec_env.single_action_space.sample() for _ in range(num_envs)])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

        if (step + 1) % 1000 == 0:
            logger.info(f"Step {step + 1}/{num_steps}")

    vec_env.close()
    logger.info("Vectorized training complete!")


def _is_game_over(game, winner):
    """
    Unified terminal-state check.

    Returns (done, reward_for_current_player) where reward is from the
    perspective of the player who just moved (passed in via `winner`).
    """
    if winner is not None:
        if winner == 0:          # draw
            return True, 0.0
        else:                    # someone won
            return True, 1.0    # caller decides sign based on whose turn it was

    if not game.get_valid_locations():   # board full, no winner declared
        return True, 0.0

    return False, 0.0


def train_self_play(episodes: int = 500):
    """
    Train two agents against each other using self-play.

    Both DQN agents learn simultaneously. Each agent always observes the board
    from its own perspective (player-1 channel = its own pieces). Rewards:
      +1  win
       0  draw / game continues
      -1  loss  (opponent won on the previous move)
    """
    logger.info("=== Self-Play Training ===")
    logger.info(f"Training two agents against each other for {episodes} episodes")

    from game.connect4 import Connect4Game

    # Use different random seeds for fair initialization
    torch.manual_seed(42)
    trainer1 = DQNTrainer(opponent_agent=None, learning_rate=1e-4, gamma=0.99)
    
    torch.manual_seed(123)  # Different seed for agent2
    trainer2 = DQNTrainer(opponent_agent=None, learning_rate=1e-4, gamma=0.99)

    logger.info("Starting self-play training...")

    agent1_wins = 0
    agent2_wins = 0
    draws = 0

    # We track per-episode outcomes for logging
    recent_results = []   # 1 = agent1 win, 2 = agent2 win, 0 = draw

    for episode in range(episodes):
        game = Connect4Game()
        done = False
        total_moves = 0

        # Keep track of each agent's last (state, action) so we can assign
        # the delayed reward when the opponent's move ends the game.
        last = {1: None, 2: None}   # {player: (state_tensor, action)}

        while not done:
            current_player = game.current_player
            trainer = trainer1 if current_player == 1 else trainer2

            valid_cols = game.get_valid_locations()

            # ── Safety guard: should never happen, but protects against an
            #    engine bug where the game continues past a full board.
            if not valid_cols:
                draws += 1
                recent_results.append(0)
                break

            # Use perspective-aware board representation
            state_tensor = trainer.board_to_tensor(game, from_player=current_player)
            action = trainer.select_action(game, training=True)

            winner = game.make_move(action)
            total_moves += 1

            done, reward_for_mover = _is_game_over(game, winner)

            if done:
                if winner and winner != 0:
                    # Current player won → opponent lost
                    opponent_player = 3 - current_player
                    opponent_trainer = trainer2 if current_player == 1 else trainer1

                    # Give +1 to the winner
                    trainer.remember(state_tensor, action, 1.0,
                                     trainer.board_to_tensor(game, from_player=current_player), True)
                    trainer.train_on_batch()

                    # Give -1 to the loser (using their last stored transition)
                    if last[opponent_player] is not None:
                        prev_state, prev_action = last[opponent_player]
                        opponent_trainer.remember(prev_state, prev_action, -1.0,
                                                  opponent_trainer.board_to_tensor(game, from_player=opponent_player), True)
                        opponent_trainer.train_on_batch()

                    if current_player == 1:
                        agent1_wins += 1
                        recent_results.append(1)
                    else:
                        agent2_wins += 1
                        recent_results.append(2)

                else:
                    # Draw: 0 reward for both
                    trainer.remember(state_tensor, action, 0.0,
                                     trainer.board_to_tensor(game, from_player=current_player), True)
                    trainer.train_on_batch()

                    opponent_player = 3 - current_player
                    opponent_trainer = trainer2 if current_player == 1 else trainer1
                    if last[opponent_player] is not None:
                        prev_state, prev_action = last[opponent_player]
                        opponent_trainer.remember(prev_state, prev_action, 0.0,
                                                  opponent_trainer.board_to_tensor(game, from_player=opponent_player), True)
                        opponent_trainer.train_on_batch()

                    draws += 1
                    recent_results.append(0)

            else:
                # Game continues: small step reward of 0, store transition
                # but don't push to replay yet — wait to see outcome
                trainer.remember(state_tensor, action, 0.0,
                                 trainer.board_to_tensor(game, from_player=current_player), False)
                trainer.train_on_batch()

            # Store this player's move so the loser can be penalised later
            last[current_player] = (state_tensor, action)

        # ── Periodic maintenance
        if (episode + 1) % 50 == 0:
            trainer1.update_target_network()
            trainer2.update_target_network()
            trainer1.decay_epsilon()
            trainer2.decay_epsilon()

        if (episode + 1) % 100 == 0:
            window = recent_results[-100:]
            w1 = window.count(1)
            w2 = window.count(2)
            dr = window.count(0)
            logger.info(
                f"Episode {episode + 1}/{episodes} | "
                f"Last 100 → Agent1 wins: {w1}  Agent2 wins: {w2}  Draws: {dr} | "
                f"Epsilon: {trainer1.epsilon:.4f}"
            )

    # ── Save both models
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trainer1.save(checkpoint_dir / f"dqn_selfplay_agent1_{timestamp}.pt")
    trainer2.save(checkpoint_dir / f"dqn_selfplay_agent2_{timestamp}.pt")

    logger.info("Self-play training complete!")
    logger.info(f"Overall → Agent1 wins: {agent1_wins}  Agent2 wins: {agent2_wins}  Draws: {draws}")

    # ── Final evaluation: test both positions
    logger.info("Evaluating trained agents against each other (greedy)...")
    eval_wins_agent1_as_p1 = 0
    eval_wins_agent1_as_p2 = 0
    eval_games = 50  # 50 games each position

    # Agent1 as Player 1
    for _ in range(eval_games):
        game = Connect4Game()
        done = False

        while not done:
            valid = game.get_valid_locations()
            if not valid:
                break

            current_player = game.current_player
            trainer = trainer1 if current_player == 1 else trainer2
            action = trainer.select_action(game, training=False)

            winner = game.make_move(action)
            done, _ = _is_game_over(game, winner)

        if winner == 1:
            eval_wins_agent1_as_p1 += 1

    # Agent1 as Player 2
    for _ in range(eval_games):
        game = Connect4Game()
        done = False

        while not done:
            valid = game.get_valid_locations()
            if not valid:
                break

            current_player = game.current_player
            trainer = trainer2 if current_player == 1 else trainer1  # Swapped: Agent2 is P1, Agent1 is P2
            action = trainer.select_action(game, training=False)

            winner = game.make_move(action)
            done, _ = _is_game_over(game, winner)

        if winner == 2:  # Agent1 wins when playing as P2 (opponent)
            eval_wins_agent1_as_p2 += 1

    win_rate_as_p1 = eval_wins_agent1_as_p1 / eval_games
    win_rate_as_p2 = eval_wins_agent1_as_p2 / eval_games
    logger.info(
        f"Agent1 as Player 1: {win_rate_as_p1:.2%} ({eval_wins_agent1_as_p1}/{eval_games})"
    )
    logger.info(
        f"Agent1 as Player 2: {win_rate_as_p2:.2%} ({eval_wins_agent1_as_p2}/{eval_games})"
    )


def train_with_stable_baselines3(algorithm: str = "dqn", timesteps: int = 100000):
    """
    Train using Stable-Baselines3 framework.

    This approach leverages state-of-the-art RL implementations with
    built-in features like prioritized experience replay, double Q-networks,
    and sophisticated exploration strategies.

    Requires stable_baselines3 to be installed.
    """
    if not HAS_SB3:
        logger.error("Stable-Baselines3 training requires: pip install stable-baselines3")
        return

    logger.info(f"=== Training with Stable-Baselines3 ({algorithm.upper()}) ===")
    logger.info(f"Total timesteps: {timesteps}")

    opponent = RuleBasedAgent(player_id=2)
    env = Connect4Env(opponent_agent=opponent, render_mode=None)

    if algorithm.lower() == "dqn":
        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-3, buffer_size=50000, learning_starts=1000,
            batch_size=32, gamma=0.99, exploration_fraction=0.5,
            exploration_initial_eps=1.0, exploration_final_eps=0.05,
            verbose=1, device="auto"
        )
    elif algorithm.lower() == "ppo":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=1e-3, n_steps=128, batch_size=64,
            n_epochs=10, gamma=0.99, verbose=1, device="auto"
        )
    elif algorithm.lower() == "a2c":
        model = A2C(
            "MlpPolicy", env,
            learning_rate=1e-3, n_steps=5, gamma=0.99,
            verbose=1, device="auto"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    checkpoint_dir = Path("checkpoints/sb3")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, timesteps // 10),
        save_path=str(checkpoint_dir),
        name_prefix=f"{algorithm}_checkpoint"
    )
    eval_callback = EvalCallback(
        env, n_eval_episodes=50,
        eval_freq=max(1, timesteps // 10),
        log_path=str(checkpoint_dir),
        best_model_save_path=str(checkpoint_dir / "best_model"),
        verbose=1
    )

    logger.info(f"Starting training with {algorithm.upper()}...")
    model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback])

    final_model_path = checkpoint_dir / f"{algorithm}_final.zip"
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    env.close()


def interactive_play(model_path: Optional[str] = None):
    """
    Play against a trained agent interactively.
    """
    logger.info("=== Interactive Play ===")

    if model_path:
        logger.info(f"Loading model from {model_path}")
        trainer = DQNTrainer(opponent_agent=None)
        trainer.load(model_path)
        agent_fn = lambda game: trainer.select_action(game, training=False)
    else:
        logger.warning("No model provided, using random opponent")
        opponent = RandomAgent(player_id=2)
        agent_fn = lambda game: opponent.select_move(game)

    env = Connect4Env(render_mode="human")
    obs, info = env.reset()

    done = False
    while not done:
        print("\nValid columns:", env._game.get_valid_locations())
        try:
            col = int(input("Enter column (0-6): "))
            if col not in env._game.get_valid_locations():
                print("Invalid column! Try again.")
                continue
        except ValueError:
            print("Invalid input! Enter a number 0-6.")
            continue

        obs, reward, terminated, truncated, info = env.step(col)
        env.render()
        done = terminated or truncated

        if done:
            print(f"\nGame Over! You {'won' if reward == 1 else 'lost' if reward == -1 else 'drew'}!")
        else:
            print("\nAgent's turn...")
            action = agent_fn(env._game)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
            if done:
                print(f"\nGame Over! You {'lost' if reward == -1 else 'drew'}!")

    env.close()


# ============================================================================ #
#  Main                                                                        #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="RL Training Workflow for Connect4")
    parser.add_argument(
        "--mode",
        choices=["fixed-opponent", "self-play", "vectorized", "sb3", "play"],
        default="fixed-opponent",
        help="Training mode"
    )
    parser.add_argument("--opponent", default="rule-based",
                        help="Opponent type (fixed-opponent mode)")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Number of timesteps (sb3 mode)")
    parser.add_argument("--algorithm", choices=["dqn", "ppo", "a2c"], default="dqn",
                        help="RL algorithm (sb3 mode)")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--model-path",
                        help="Path to trained model (play mode)")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Connect4 RL Training Workflow")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 70)

    if args.mode == "fixed-opponent":
        train_fixed_opponent(opponent_name=args.opponent, episodes=args.episodes)
    elif args.mode == "self-play":
        train_self_play(episodes=args.episodes)
    elif args.mode == "vectorized":
        train_vectorized_envs(num_envs=args.num_envs, episodes=args.episodes)
    elif args.mode == "sb3":
        train_with_stable_baselines3(algorithm=args.algorithm, timesteps=args.timesteps)
    elif args.mode == "play":
        interactive_play(model_path=args.model_path)


if __name__ == "__main__":
    main()