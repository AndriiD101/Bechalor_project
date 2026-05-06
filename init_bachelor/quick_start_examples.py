"""
Quick Start Examples for Connect4 RL Training

This file demonstrates the most common usage patterns for training agents.
Run individual examples to see the RL training workflow in action.
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

from rl_training_workflow import (
    DQNTrainer, train_fixed_opponent, train_with_stable_baselines3,
    interactive_play
)
from agents.rule_based_agent import RuleBasedAgent
from agents.random_agent import RandomAgent
from game.connect4_env import Connect4Env


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================ #
#  Example 1: Basic DQN Training (5 minutes)                                 #
# ============================================================================ #

def example_1_basic_training():
    """
    Train a simple DQN agent against the Rule-Based opponent.
    
    This is the quickest way to see the RL training in action.
    Takes ~5 minutes for 200 episodes on CPU.
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Basic DQN Training")
    logger.info("="*70)
    logger.info("Training a DQN agent to beat the Rule-Based opponent...")
    logger.info("This will take ~5 minutes on CPU\n")
    
    # Create opponent
    opponent = RuleBasedAgent(player_id=2)
    
    # Create trainer
    trainer = DQNTrainer(opponent_agent=opponent)
    
    # Train for a small number of episodes
    logger.info("Starting training...")
    rewards = trainer.train(num_episodes=200, target_update_freq=20, eval_freq=20)
    
    # Save the trained model
    checkpoint_path = Path("checkpoints/example1_basic_dqn.pt")
    trainer.save(checkpoint_path)
    logger.info(f"\n✓ Model saved to {checkpoint_path}")
    
    # Evaluate
    logger.info("\nEvaluating trained agent...")
    win_rate = trainer.evaluate(num_games=50, opponent=opponent)
    logger.info(f"✓ Evaluation complete! Win rate: {win_rate:.1%}")


# ============================================================================ #
#  Example 2: Training with Different Opponents (10 minutes)                 #
# ============================================================================ #

def example_2_compare_opponents():
    """
    Train against different opponents and compare difficulty levels.
    
    Shows how the same training algorithm performs against:
    - Random (easy)
    - Rule-Based (medium)
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Training Against Different Opponents")
    logger.info("="*70)
    
    opponents_config = [
        ("random", RandomAgent(player_id=2), 100),
        ("rule-based", RuleBasedAgent(player_id=2), 200),
    ]
    
    for opponent_name, opponent, episodes in opponents_config:
        logger.info(f"\nTraining against {opponent_name.upper()} ({episodes} episodes)...")
        
        trainer = DQNTrainer(opponent_agent=opponent)
        trainer.train(num_episodes=episodes, target_update_freq=20, eval_freq=50)
        
        # Save
        checkpoint_path = Path("checkpoints") / f"example2_vs_{opponent_name}.pt"
        trainer.save(checkpoint_path)
        
        # Evaluate
        win_rate = trainer.evaluate(num_games=30, opponent=opponent)
        logger.info(f"  ✓ Win rate vs {opponent_name}: {win_rate:.1%}")


# ============================================================================ #
#  Example 3: Play Against Trained Agent (interactive)                       #
# ============================================================================ #

def example_3_interactive_play():
    """
    Play an interactive game against a trained agent.
    
    Load a previously trained model and play against it.
    You control Player 1 (blue), agent controls Player 2 (red).
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Interactive Play Against Trained Agent")
    logger.info("="*70)
    
    # Check if a trained model exists
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        logger.warning("No checkpoints directory found.")
        logger.info("Train a model first using Example 1 or 2")
        return
    
    existing_models = list(checkpoint_dir.glob("*.pt"))
    if not existing_models:
        logger.warning("No trained models found in checkpoints/")
        logger.info("Train a model first using Example 1 or 2")
        return
    
    # Try to use the most recent model
    model_path = sorted(existing_models)[-1]
    logger.info(f"Using model: {model_path.name}")
    
    # Start interactive play
    logger.info("\nStarting interactive game...")
    logger.info("You are Player 1 (blue). Enter column numbers 0-6 for your moves.\n")
    
    interactive_play(model_path=str(model_path))


# ============================================================================ #
#  Example 4: Evaluate Training Progress (for longer training)              #
# ============================================================================ #

def example_4_training_with_eval():
    """
    Train with periodic evaluation to monitor learning progress.
    
    Demonstrates how to save the best model during training
    and track improvement over time.
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Training with Periodic Evaluation")
    logger.info("="*70)
    
    opponent = RuleBasedAgent(player_id=2)
    trainer = DQNTrainer(opponent_agent=opponent)
    
    logger.info("Training with evaluation every 50 episodes...")
    logger.info("This demonstrates tracking improvement over time\n")
    
    best_win_rate = 0.0
    best_model_path = Path("checkpoints/example4_best_model.pt")
    
    # Train in phases to show improvement
    for phase in range(3):
        episodes_this_phase = 150
        logger.info(f"\n--- Phase {phase + 1}/3 ({episodes_this_phase} episodes) ---")
        
        trainer.train(num_episodes=episodes_this_phase, target_update_freq=30, eval_freq=50)
        
        # Evaluate and save best
        win_rate = trainer.evaluate(num_games=50, opponent=opponent)
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            trainer.save(best_model_path)
            logger.info(f"  ✓ New best model! Win rate: {win_rate:.1%}")
        else:
            logger.info(f"  ! Win rate: {win_rate:.1%} (best so far: {best_win_rate:.1%})")
    
    logger.info(f"\n✓ Best model saved to {best_model_path}")
    logger.info(f"✓ Best win rate achieved: {best_win_rate:.1%}")


# ============================================================================ #
#  Example 5: Demonstrating the Gymnasium Environment                       #
# ============================================================================ #

def example_5_gymnasium_api():
    """
    Show how to use the Connect4Env directly with the Gymnasium API.
    
    This demonstrates the standard environment interface that any
    RL algorithm can use.
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 5: Using the Gymnasium Environment Directly")
    logger.info("="*70)
    
    logger.info("\nDemonstrating the standard Gymnasium API:")
    logger.info("  1. Create environment")
    logger.info("  2. Reset for new episode")
    logger.info("  3. Step through game")
    logger.info("  4. Use observations and rewards for training\n")
    
    # Create environment
    opponent = RuleBasedAgent(player_id=2)
    env = Connect4Env(opponent_agent=opponent, render_mode="ansi")
    
    logger.info("Environment created!")
    logger.info(f"  Observation space: {env.observation_space}")
    logger.info(f"  Action space: {env.action_space}")
    
    # Reset
    logger.info("\nResetting environment...")
    obs, info = env.reset()
    logger.info(f"  Initial observation shape: {obs.shape}")
    logger.info(f"  Info: {info}")
    
    # Play one episode
    logger.info("\nPlaying one episode (agent makes random moves)...")
    done = False
    step_count = 0
    
    while not done:
        # Agent chooses random action
        action = env.action_space.sample()
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        logger.info(
            f"  Step {step_count}: action={action}, reward={reward}, "
            f"done={done}, current_player={info['current_player']}"
        )
        
        if step_count >= 5:  # Just show first 5 steps
            logger.info("  ... (continuing game)")
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
            break
    
    logger.info(f"\n✓ Episode finished after {step_count} steps")
    logger.info(f"✓ Final reward: {reward}")
    logger.info(f"✓ Winner: {info['winner'] if info['winner'] else 'Draw'}")
    
    env.close()


# ============================================================================ #
#  Example 6: Simple Baseline Comparison (advanced)                         #
# ============================================================================ #

def example_6_baseline_comparison():
    """
    Compare trained agent's performance against fixed agents.
    
    Demonstrates how to use trained models to benchmark against
    other strategies.
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 6: Agent Comparison")
    logger.info("="*70)
    
    # First, train a model
    logger.info("\nTraining a DQN agent (100 episodes)...")
    opponent = RuleBasedAgent(player_id=2)
    trainer = DQNTrainer(opponent_agent=opponent)
    trainer.train(num_episodes=100, target_update_freq=20, eval_freq=50)
    
    # Now evaluate against different opponents
    logger.info("\n" + "-"*70)
    logger.info("Evaluating trained agent against different opponents:")
    logger.info("-"*70)
    
    test_opponents = [
        ("Random", RandomAgent(player_id=2)),
        ("Rule-Based", RuleBasedAgent(player_id=2)),
    ]
    
    for name, test_opponent in test_opponents:
        win_rate = trainer.evaluate(num_games=50, opponent=test_opponent)
        logger.info(f"vs {name:12s}: {win_rate:5.1%}")
    
    logger.info("-"*70)


# ============================================================================ #
#  Main Menu                                                                  #
# ============================================================================ #

def main():
    """Interactive menu to run examples."""
    
    examples = {
        "1": ("Basic DQN Training (5 min)", example_1_basic_training),
        "2": ("Compare Different Opponents (10 min)", example_2_compare_opponents),
        "3": ("Play Interactive Game", example_3_interactive_play),
        "4": ("Training with Evaluation (15 min)", example_4_training_with_eval),
        "5": ("Gymnasium API Demo", example_5_gymnasium_api),
        "6": ("Agent Comparison (8 min)", example_6_baseline_comparison),
    }
    
    print("\n" + "="*70)
    print("Connect4 RL Training - Quick Start Examples")
    print("="*70 + "\n")
    
    print("Choose an example to run:\n")
    for key, (description, _) in examples.items():
        print(f"  [{key}] {description}")
    print(f"  [q] Quit\n")
    
    choice = input("Enter your choice: ").strip().lower()
    
    if choice in examples:
        _, example_fn = examples[choice]
        example_fn()
    elif choice == "q":
        print("Goodbye!")
    else:
        print("Invalid choice. Please try again.")
        main()


if __name__ == "__main__":
    import sys
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples_by_arg = {
            "1": example_1_basic_training,
            "2": example_2_compare_opponents,
            "3": example_3_interactive_play,
            "4": example_4_training_with_eval,
            "5": example_5_gymnasium_api,
            "6": example_6_baseline_comparison,
        }
        if example_num in examples_by_arg:
            examples_by_arg[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print(f"Available: {', '.join(examples_by_arg.keys())}")
    else:
        # Interactive menu
        main()
