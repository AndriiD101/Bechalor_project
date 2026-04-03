"""
train_dqn.py
============
Trains the DQNAgent (reinforcement_agent.py) to play Connect-4.

Key design decisions:
  - Double DQN: a frozen target network stabilises Q-value targets.
  - Replay buffer: random sampling breaks temporal correlations.
  - Opponent curriculum: starts vs Random, graduates to RuleBased → MCTS → Self-play.
  - Reward shaping: win/loss/draw + small intermediate signals.
  - Periodic checkpointing + best-model saving.

Run:
    python train_dqn.py
"""

import random
import collections
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.connect4 import Connect4Game
from agents.reinforcement_agent import DQNAgent, Connect4Net
from agents.random_agent import RandomAgent
from agents.mcts_agent import MCTSAgent
from agents.rule_based_agent import RuleBasedAgent


# ================================================================== #
#  Hyper-parameters — tweak these                                     #
# ================================================================== #

EPISODES          = 50_000   # Total training games
BATCH_SIZE        = 128      # Samples per gradient update
REPLAY_CAPACITY   = 50_000   # Max transitions stored
GAMMA             = 0.97     # Discount factor
LR                = 1e-4     # Adam learning rate
TARGET_UPDATE     = 500      # Sync target net every N episodes
TRAIN_EVERY       = 4        # Learn every N steps (not every step)

# Epsilon schedule
EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY_STEPS   = 30_000   # Linear decay over this many episodes

# Opponent curriculum thresholds (win-rate over last 500 games)
CURRICULUM_THRESHOLD = 0.60  # Graduate to next opponent at this win-rate

# Checkpointing
SAVE_DIR          = "checkpoints"
SAVE_EVERY        = 2_000    # Save periodic checkpoint every N episodes
EVAL_EVERY        = 1_000    # Evaluate vs Random every N episodes
EVAL_GAMES        = 100      # Games per evaluation

# Rewards
R_WIN             =  1.0
R_LOSS            = -1.0
R_DRAW            =  0.2     # Small positive — draw is better than loss
R_STEP            = -0.01    # Tiny penalty to encourage faster wins


# ================================================================== #
#  Replay Buffer                                                       #
# ================================================================== #

Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ================================================================== #
#  Board encoding (reuses DQNAgent logic)                             #
# ================================================================== #

def board_to_tensor(board: np.ndarray, player_id: int, valid_moves: list, device):
    opponent_id = 2 if player_id == 1 else 1
    my_layer    = (board == player_id).astype(np.float32)
    opp_layer   = (board == opponent_id).astype(np.float32)
    valid_layer = np.zeros_like(my_layer)
    for col in valid_moves:
        valid_layer[:, col] = 1.0
    state = np.stack([my_layer, opp_layer, valid_layer], axis=0)
    return torch.tensor(state).unsqueeze(0).to(device)


# ================================================================== #
#  Opponent helpers                                                    #
# ================================================================== #

def opponent_move(opponent, game: Connect4Game) -> int:
    """Get opponent's move; fallback to random if it errors."""
    try:
        move = opponent.select_move(game)
        if move is None or move == -1:
            moves = [c for c in range(game.column_count) if game.is_valid_location(game.board, c)]
            return random.choice(moves) if moves else -1
        return move
    except Exception:
        moves = [c for c in range(game.column_count) if game.is_valid_location(game.board, c)]
        return random.choice(moves) if moves else -1


# ================================================================== #
#  Single training episode                                             #
# ================================================================== #

def run_episode(
    agent: DQNAgent,
    opponent,
    replay: ReplayBuffer,
    device,
    step_counter: list,   # mutable int wrapper [count]
) -> str:
    """
    Play one full game. Agent and opponent alternate; who goes first is
    randomised each episode so the agent learns both sides of the board.
    Returns "win" | "loss" | "draw".
    """
    game = Connect4Game()

    # Randomly decide who goes first so agent learns both positions
    agent_is_p1 = random.random() < 0.5
    agent_pid   = 1 if agent_is_p1 else 2
    agent.player_id   = agent_pid
    agent.opponent_id = 2 if agent_pid == 1 else 1

    outcome     = "draw"
    # Saved state/action from the agent's last move (needed to push the
    # intermediate transition once we see what the opponent does).
    last_state_t = None
    last_action  = None

    while True:
        valid = [c for c in range(game.column_count)
                 if game.is_valid_location(game.board, c)]
        if not valid:
            break

        # ------------------------------------------------------------------ #
        #  Agent's turn                                                        #
        # ------------------------------------------------------------------ #
        if game.current_player == agent_pid:
            state_t = board_to_tensor(game.board, agent_pid, valid, device)

            # Epsilon-greedy action
            if random.random() < agent.epsilon:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    q = agent.policy_net(state_t).cpu().numpy().squeeze()
                masked = np.full(7, -1e9)
                for c in valid:
                    masked[c] = q[c]
                action = int(np.argmax(masked))

            success, row = game.make_move(action)
            if not success:
                break

            step_counter[0] += 1

            # --- Terminal: agent wins ---
            if game.check_winner(row, action, agent_pid):
                next_state_t = board_to_tensor(game.board, agent_pid, [], device)
                replay.push(state_t, action, R_WIN, next_state_t, True)
                outcome = "win"
                break

            # --- Terminal: draw after agent's move ---
            if game.check_draw():
                next_state_t = board_to_tensor(game.board, agent_pid, [], device)
                replay.push(state_t, action, R_DRAW, next_state_t, True)
                outcome = "draw"
                break

            # Save for intermediate transition after opponent responds
            last_state_t = state_t
            last_action  = action

            game.switch_player()

        # ------------------------------------------------------------------ #
        #  Opponent's turn                                                     #
        # ------------------------------------------------------------------ #
        else:
            opp_valid = [c for c in range(game.column_count)
                         if game.is_valid_location(game.board, c)]
            if not opp_valid:
                break

            # Capture current player id BEFORE make_move potentially mutates it
            opp_pid_now = game.current_player

            if hasattr(opponent, "player_id"):
                opponent.player_id   = opp_pid_now
                opponent.opponent_id = agent_pid

            opp_action = opponent_move(opponent, game)
            if opp_action == -1:
                break

            success, opp_row = game.make_move(opp_action)
            if not success:
                break

            # --- Terminal: opponent wins ---
            # BUG FIX 3: use opp_pid_now (captured before make_move),
            # NOT game.current_player which may be mutated after make_move.
            if game.check_winner(opp_row, opp_action, opp_pid_now):
                if last_state_t is not None:
                    next_state_t = board_to_tensor(game.board, agent_pid, [], device)
                    replay.push(last_state_t, last_action, R_LOSS, next_state_t, True)
                outcome = "loss"
                break

            # --- Terminal: draw after opponent's move ---
            if game.check_draw():
                if last_state_t is not None:
                    next_state_t = board_to_tensor(game.board, agent_pid, [], device)
                    replay.push(last_state_t, last_action, R_DRAW, next_state_t, True)
                outcome = "draw"
                break

            # --- Non-terminal: push intermediate transition for agent's move ---
            if last_state_t is not None:
                next_valid   = [c for c in range(game.column_count)
                                if game.is_valid_location(game.board, c)]
                next_state_t = board_to_tensor(game.board, agent_pid, next_valid, device)
                replay.push(last_state_t, last_action, R_STEP, next_state_t, False)
                last_state_t = None
                last_action  = None

            game.switch_player()

    return outcome


# ================================================================== #
#  Learning step                                                       #
# ================================================================== #

def learn(
    policy_net: Connect4Net,
    target_net: Connect4Net,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    device,
):
    if len(replay) < BATCH_SIZE:
        return None

    batch = replay.sample(BATCH_SIZE)
    batch = Transition(*zip(*batch))

    states      = torch.cat(batch.state)
    actions     = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards     = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_states = torch.cat(batch.next_state)
    dones       = torch.tensor(batch.done, dtype=torch.float32, device=device)

    # Current Q-values
    current_q = policy_net(states).gather(1, actions).squeeze(1)

    # Double DQN target: action chosen by policy net, value from target net
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        next_q       = target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q     = rewards + GAMMA * next_q * (1 - dones)

    loss = nn.SmoothL1Loss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
    optimizer.step()

    return loss.item()


# ================================================================== #
#  Evaluation                                                          #
# ================================================================== #

def evaluate(agent: DQNAgent, device, stage_name: str, n_games: int = EVAL_GAMES) -> float:
    """
    Win-rate vs the *current* curriculum opponent over n_games.
    Using the same opponent the agent is training against gives a meaningful
    signal at every stage instead of always testing vs RuleBased (which
    would show 100% loss while the agent is still on Random).
    """
    wins   = 0
    losses = 0
    draws  = 0
    saved_eps = agent.epsilon
    agent.epsilon = 0.0   # fully greedy during eval

    for _ in range(n_games):
        game = Connect4Game()

        # Randomly assign sides so we measure both P1 and P2 performance
        agent_is_p1 = random.random() < 0.5
        agent_pid   = 1 if agent_is_p1 else 2
        opp_pid     = 2 if agent_is_p1 else 1

        # BUG FIX 1: instantiate the *current stage* opponent, not always
        # RuleBasedAgent — the original code evaluated vs RuleBased even
        # while training vs Random, producing 100% loss forever.
        if stage_name == "Random":
            from agents.random_agent import RandomAgent
            opponent = RandomAgent(player_id=opp_pid)
        # elif stage_name == "RuleBased":
        #     opponent = RuleBasedAgent(player_id=opp_pid)
        elif stage_name == "MCTS":
            opponent = MCTSAgent(player_id=opp_pid)
        # else:  # Self — evaluate vs RuleBased as a stable reference
        #     opponent = RuleBasedAgent(player_id=opp_pid)

        agent.player_id   = agent_pid
        agent.opponent_id = opp_pid

        result = "draw"

        while True:
            current = game.current_player

            valid = [c for c in range(game.column_count)
                     if game.is_valid_location(game.board, c)]
            if not valid:
                break

            if current == agent_pid:
                move = agent.select_move(game)
                if move is None:
                    break
                success, row = game.make_move(move)
                if not success:
                    break
                if game.check_winner(row, move, agent_pid):
                    result = "win"
                    break
                if game.check_draw():
                    result = "draw"
                    break

            else:
                opponent.player_id   = current
                opponent.opponent_id = agent_pid
                move = opponent.select_move(game)
                if move is None or move == -1:
                    break
                success, row = game.make_move(move)
                if not success:
                    break
                # BUG FIX 2: check winner with opp_pid (a fixed variable),
                # NOT game.current_player — make_move may mutate current_player
                # depending on the Connect4Game implementation, so reading it
                # after the move is unreliable.
                if game.check_winner(row, move, opp_pid):
                    result = "loss"
                    break
                if game.check_draw():
                    result = "draw"
                    break

            game.switch_player()

        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            draws += 1

    agent.epsilon = saved_eps

    win_rate  = wins   / n_games
    loss_rate = losses / n_games
    draw_rate = draws  / n_games

    print(
        f"  Eval vs {stage_name:<9} ({n_games} games): "
        f"Win {win_rate:.0%}  |  Loss {loss_rate:.0%}  |  Draw {draw_rate:.0%}"
    )

    return win_rate


# ================================================================== #
#  Main training loop                                                  #
# ================================================================== #

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Networks
    policy_net = Connect4Net().to(device)
    target_net = Connect4Net().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay    = ReplayBuffer(REPLAY_CAPACITY)

    # Agent wrapper (policy_net shared by reference)
    agent = DQNAgent(player_id=1)
    agent.policy_net = policy_net
    agent.epsilon    = EPS_START

    # Curriculum: Random → RuleBased → MCTS → Self
    curriculum_stages = ["Random", "MCTS", "Self"]
    stage_idx  = 0
    stage_name = curriculum_stages[stage_idx]
    recent_outcomes = collections.deque(maxlen=500)

    def make_opponent(name):
        if name == "Random":
            return RandomAgent(player_id=2)
        # elif name == "RuleBased":
        #     return RuleBasedAgent(player_id=2)
        elif name == "MCTS":
            return MCTSAgent(player_id=2)
        else:
            # BUG FIX: move the snapshot network to the same device so
            # forward passes don't crash with a device mismatch.
            snap = DQNAgent(player_id=2)
            snap.policy_net.load_state_dict(policy_net.state_dict())
            snap.policy_net.to(device)
            snap.epsilon = 0.05
            return snap

    opponent = make_opponent(stage_name)

    step_counter  = [0]
    best_win_rate = 0.0
    total_loss    = 0.0
    loss_count    = 0
    t0            = time.time()

    print(f"\nStarting training — {EPISODES} episodes")
    print(f"Stage 0: vs {stage_name}  (Random → MCTS → Self)\n")

    for episode in range(1, EPISODES + 1):

        # --- Epsilon decay (linear) ---
        agent.epsilon = max(
            EPS_END,
            EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_STEPS)
        )

        # --- Play one episode ---
        outcome = run_episode(agent, opponent, replay, device, step_counter)
        recent_outcomes.append(outcome)

        # --- Learn ---
        # BUG FIX 4: gate learning on episode count, not step_counter.
        # step_counter increments only on the agent's moves (~half of all
        # moves), so TRAIN_EVERY=4 on steps meant learning almost every
        # episode anyway but with an unpredictable cadence. Gating on
        # episodes gives a clean, predictable update frequency.
        if episode % TRAIN_EVERY == 0:
            loss = learn(policy_net, target_net, optimizer, replay, device)
            if loss is not None:
                total_loss += loss
                loss_count += 1

        # --- Sync target network ---
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # --- Curriculum check ---
        if len(recent_outcomes) == 500:
            win_rate = recent_outcomes.count("win") / 500
            if win_rate >= CURRICULUM_THRESHOLD and stage_idx < len(curriculum_stages) - 1:
                stage_idx  += 1
                stage_name  = curriculum_stages[stage_idx]
                opponent    = make_opponent(stage_name)
                recent_outcomes.clear()
                print(
                    f"\n  >>> Graduated to stage {stage_idx}: vs {stage_name}"
                    f"  (win-rate was {win_rate:.0%})\n"
                )

            # For self-play, refresh the snapshot periodically
            if stage_name == "Self" and episode % 2000 == 0:
                opponent = make_opponent("Self")

        # --- Periodic checkpoint ---
        if episode % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"dqn_ep{episode}.pth")
            torch.save(policy_net.state_dict(), ckpt_path)

        # --- Evaluation + best model ---
        if episode % EVAL_EVERY == 0:
            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            elapsed  = time.time() - t0
            print(
                f"Ep {episode:>6} | stage={stage_name:<9} | "
                f"eps={agent.epsilon:.3f} | loss={avg_loss:.4f} | "
                f"buf={len(replay):>6} | t={elapsed:.0f}s"
            )
            # evaluate() now prints win/loss/draw breakdown internally
            win_rate = evaluate(agent, device, stage_name)
            total_loss = 0.0
            loss_count = 0

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = "dqn_connect4_best.pth"
                torch.save(policy_net.state_dict(), best_path)
                print(f"  ★ New best model saved ({win_rate:.0%}) → {best_path}")

    # --- Final save ---
    final_path = os.path.join(SAVE_DIR, "dqn_final.pth")
    torch.save(policy_net.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best win-rate achieved: {best_win_rate:.0%}")


if __name__ == "__main__":
    train()