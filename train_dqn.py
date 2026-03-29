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
from agents.rule_based_agent import RuleBasedAgent


# ================================================================== #
#  Hyper-parameters                                                    #
# ================================================================== #

EPISODES          = 50_000
BATCH_SIZE        = 128
REPLAY_CAPACITY   = 50_000
GAMMA             = 0.97
LR                = 1e-4
TARGET_UPDATE     = 500      # Sync target net every N episodes
TRAIN_EVERY       = 4        # Learn every N environment steps

# Epsilon schedule (linear decay)
EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY_STEPS   = 30_000

# Opponent curriculum
CURRICULUM_THRESHOLD = 0.60  # Graduate at this win-rate over last 500 games

# Checkpointing
SAVE_DIR          = "checkpoints"
SAVE_EVERY        = 2_000
EVAL_EVERY        = 1_000
EVAL_GAMES        = 100

# Rewards
R_WIN             =  1.0
R_LOSS            = -1.0
R_DRAW            =  0.2
R_STEP            = -0.01

# FIX 1 — PER parameters
PER_ALPHA         = 0.6      # How much to prioritise by TD error (0 = uniform)
PER_BETA_START    = 0.4      # IS weight correction (annealed to 1.0)
PER_BETA_STEPS    = 30_000   # Steps over which beta is annealed
PER_EPS           = 1e-6     # Small constant to avoid zero priority

# FIX 2 — LR schedule
LR_T0             = 5_000    # Episodes per cosine cycle
LR_T_MULT         = 2        # Cycle length multiplier after each restart


# ================================================================== #
#  FIX 1: Prioritised Experience Replay                               #
# ================================================================== #

class PrioritisedReplayBuffer:
    """
    Sum-tree backed prioritised replay buffer.

    Stores raw (6,7) int8 numpy boards (FIX 4) instead of pre-encoded
    tensors, reducing memory by ~12×.

    Transitions are sampled proportionally to priority^alpha.
    Importance-sampling weights (beta) correct the update bias.
    """

    def __init__(self, capacity: int, alpha: float = PER_ALPHA):
        self.capacity  = capacity
        self.alpha     = alpha
        self._data: list               = [None] * capacity
        self._priorities: np.ndarray   = np.zeros(capacity, dtype=np.float32)
        self._ptr      = 0
        self._size     = 0
        self._max_prio = 1.0

    def push(self, state_board, action, reward, next_board, done,
             player_id, valid_next):
        """
        state_board / next_board: raw (6,7) int8 numpy array.
        player_id: the agent's player id (needed to re-encode at sample time).
        valid_next: list of valid column indices for next_board.
        """
        idx = self._ptr
        self._data[idx] = (state_board, action, reward, next_board,
                           done, player_id, valid_next)
        self._priorities[idx] = self._max_prio
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        """
        Returns (indices, samples, is_weights).
        is_weights are importance-sampling correction factors.
        """
        prios = self._priorities[:self._size]
        probs = (prios ** self.alpha)
        probs /= probs.sum()

        indices = np.random.choice(self._size, batch_size,
                                   replace=False, p=probs)
        samples = [self._data[i] for i in indices]

        # Importance-sampling weights
        total    = self._size
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()   # normalise so max weight = 1

        return indices, samples, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray,
                          td_errors: np.ndarray) -> None:
        prios = np.abs(td_errors) + PER_EPS
        self._priorities[indices] = prios
        self._max_prio = max(self._max_prio, prios.max())

    def __len__(self):
        return self._size


# ================================================================== #
#  FIX 3+4: Pre-allocated batch tensors + numpy board encoding        #
# ================================================================== #

def _encode_board(board: np.ndarray, player_id: int,
                  valid_moves: list) -> np.ndarray:
    """
    Convert a raw (6,7) int8 board to a (3,6,7) float32 numpy array.
    Encoding is deferred to sample time to save memory in the buffer.
    """
    opponent_id  = 2 if player_id == 1 else 1
    board_flip   = np.flipud(board)
    my_layer     = (board_flip == player_id).astype(np.float32)
    opp_layer    = (board_flip == opponent_id).astype(np.float32)
    valid_layer  = np.zeros((6, 7), dtype=np.float32)
    for col in valid_moves:
        valid_layer[:, col] = 1.0
    return np.stack([my_layer, opp_layer, valid_layer])   # (3,6,7)


class BatchBuffer:
    """
    Pre-allocated CPU tensors reused across learn() calls (FIX 3).
    Only one .to(device) transfer per batch instead of many small ones.
    """
    def __init__(self, batch_size: int):
        self.states      = torch.zeros(batch_size, 3, 6, 7)
        self.next_states = torch.zeros(batch_size, 3, 6, 7)
        self.actions     = torch.zeros(batch_size, dtype=torch.long)
        self.rewards     = torch.zeros(batch_size)
        self.dones       = torch.zeros(batch_size)
        self.weights     = torch.zeros(batch_size)

    def fill(self, samples, is_weights):
        for i, (s_board, action, reward, ns_board,
                done, pid, valid_next) in enumerate(samples):
            self.states[i]      = torch.from_numpy(_encode_board(s_board, pid, []))
            self.next_states[i] = torch.from_numpy(_encode_board(ns_board, pid, valid_next))
            self.actions[i]     = action
            self.rewards[i]     = reward
            self.dones[i]       = float(done)
            self.weights[i]     = is_weights[i]


# ================================================================== #
#  Opponent helpers                                                    #
# ================================================================== #

def opponent_move(opponent, game: Connect4Game) -> int:
    try:
        move = opponent.select_move(game)
        if move is None or move == -1:
            valid = [c for c in range(game.column_count)
                     if game.is_valid_location(game.board, c)]
            return random.choice(valid) if valid else -1
        return move
    except Exception:
        valid = [c for c in range(game.column_count)
                 if game.is_valid_location(game.board, c)]
        return random.choice(valid) if valid else -1


# ================================================================== #
#  Single training episode                                             #
# ================================================================== #

def run_episode(
    agent: DQNAgent,
    opponent,
    replay: PrioritisedReplayBuffer,
    device,
    step_counter: list,
) -> str:
    """
    Play one full game, storing raw board transitions.
    Returns "win" | "loss" | "draw".
    """
    game = Connect4Game()

    agent_is_p1 = random.random() < 0.5
    agent_pid   = 1 if agent_is_p1 else 2
    agent.player_id   = agent_pid
    agent.opponent_id = 2 if agent_pid == 1 else 1

    outcome = "draw"

    while True:
        valid = [c for c in range(game.column_count)
                 if game.is_valid_location(game.board, c)]
        if not valid:
            break

        if game.current_player == agent_pid:
            # FIX 4: store raw board copy — encode later at sample time
            state_board = game.board.copy()

            # Epsilon-greedy action
            if random.random() < agent.epsilon:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    state_t  = agent._board_to_tensor(game.board, valid)
                    q        = agent.policy_net(state_t).cpu().numpy().squeeze()
                valid_mask = np.zeros(7, dtype=bool)
                for c in valid:
                    valid_mask[c] = True
                action = int(np.argmax(np.where(valid_mask, q, -1e9)))

            success, row = game.make_move(action)
            if not success:
                break

            step_counter[0] += 1

            if game.check_winner(row, action, agent_pid):
                replay.push(state_board, action, R_WIN,
                            game.board.copy(), True, agent_pid, [])
                outcome = "win"
                break

            if game.check_draw():
                replay.push(state_board, action, R_DRAW,
                            game.board.copy(), True, agent_pid, [])
                outcome = "draw"
                break

            game.switch_player()

            # Opponent's turn
            opp_valid = [c for c in range(game.column_count)
                         if game.is_valid_location(game.board, c)]
            if not opp_valid:
                break

            if hasattr(opponent, "player_id"):
                opponent.player_id   = game.current_player
                opponent.opponent_id = agent_pid

            opp_action = opponent_move(opponent, game)
            if opp_action == -1:
                break

            success, opp_row = game.make_move(opp_action)
            if not success:
                break

            if game.check_winner(opp_row, opp_action, game.current_player):
                replay.push(state_board, action, R_LOSS,
                            game.board.copy(), True, agent_pid, [])
                outcome = "loss"
                break

            if game.check_draw():
                replay.push(state_board, action, R_DRAW,
                            game.board.copy(), True, agent_pid, [])
                outcome = "draw"
                break

            game.switch_player()

            # Intermediate step transition
            next_valid = [c for c in range(game.column_count)
                          if game.is_valid_location(game.board, c)]
            replay.push(state_board, action, R_STEP,
                        game.board.copy(), False, agent_pid, next_valid)

        else:
            # Agent is P2 — opponent moves first
            if hasattr(opponent, "player_id"):
                opponent.player_id   = game.current_player
                opponent.opponent_id = agent_pid

            opp_action = opponent_move(opponent, game)
            if opp_action == -1:
                break

            success, opp_row = game.make_move(opp_action)
            if not success:
                break

            if game.check_winner(opp_row, opp_action, game.current_player):
                outcome = "loss"
                break

            if game.check_draw():
                outcome = "draw"
                break

            game.switch_player()

    return outcome


# ================================================================== #
#  Learning step                                                       #
# ================================================================== #

def learn(
    policy_net: Connect4Net,
    target_net: Connect4Net,
    optimizer: optim.Optimizer,
    replay: PrioritisedReplayBuffer,
    batch_buf: BatchBuffer,
    device,
    beta: float,
) -> float | None:
    if len(replay) < BATCH_SIZE:
        return None

    # FIX 1: prioritised sampling
    indices, samples, is_weights = replay.sample(BATCH_SIZE, beta)

    # FIX 3: fill pre-allocated tensors in-place
    batch_buf.fill(samples, is_weights)

    states      = batch_buf.states.to(device)
    next_states = batch_buf.next_states.to(device)
    actions     = batch_buf.actions.unsqueeze(1).to(device)
    rewards     = batch_buf.rewards.to(device)
    dones       = batch_buf.dones.to(device)
    weights     = batch_buf.weights.to(device)

    # FIX 5: switch to train mode only for the gradient step
    policy_net.train()

    current_q = policy_net(states).gather(1, actions).squeeze(1)

    # Double DQN target
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        next_q       = target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q     = rewards + GAMMA * next_q * (1 - dones)

    td_errors = (current_q - target_q).detach().cpu().numpy()

    # FIX 1: IS-weighted Huber loss
    element_loss = nn.SmoothL1Loss(reduction="none")(current_q, target_q)
    loss = (weights * element_loss).mean()

    # FIX 7: set_to_none=True is faster than zeroing gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
    optimizer.step()

    # FIX 5: back to eval mode immediately
    policy_net.eval()

    # FIX 1: update priorities with fresh TD errors
    replay.update_priorities(indices, td_errors)

    return loss.item()


# ================================================================== #
#  FIX 6: Evaluation against multiple opponents                       #
# ================================================================== #

def evaluate(agent: DQNAgent, device,
             n_games: int = EVAL_GAMES) -> dict[str, float]:
    """
    Win-rate vs Random and RuleBased (FIX 6).
    Returns {"random": wr, "rule": wr}.
    """
    saved_eps   = agent.epsilon
    agent.epsilon = 0.0   # greedy during eval

    results = {}
    for opp_name, opp_cls, opp_pid in [
        ("random", RandomAgent,    2),
        ("rule",   RuleBasedAgent, 2),
    ]:
        opponent = opp_cls(player_id=opp_pid)
        wins = 0

        for _ in range(n_games):
            game = Connect4Game()
            agent.player_id   = 1
            agent.opponent_id = 2

            while True:
                if game.current_player == 1:
                    move = agent.select_move(game)
                    if move is None or move == -1:
                        break
                    success, row = game.make_move(move)
                    if not success:
                        break
                    if game.check_winner(row, move, 1):
                        wins += 1
                        break
                    if game.check_draw():
                        break
                    game.switch_player()
                else:
                    move = opponent.select_move(game)
                    if move is None or move == -1:
                        break
                    success, row = game.make_move(move)
                    if not success:
                        break
                    if game.check_winner(row, move, 2):
                        break
                    if game.check_draw():
                        break
                    game.switch_player()

        results[opp_name] = wins / n_games

    agent.epsilon = saved_eps
    return results


# ================================================================== #
#  Main training loop                                                  #
# ================================================================== #

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on: {device}")

    # Networks
    policy_net = Connect4Net().to(device)
    target_net = Connect4Net().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    policy_net.eval()   # FIX 5: default eval; train() only during learn()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # FIX 2: cosine LR schedule with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=LR_T0, T_mult=LR_T_MULT, eta_min=LR * 0.01
    )

    # FIX 1+4: prioritised replay with raw board storage
    replay    = PrioritisedReplayBuffer(REPLAY_CAPACITY)
    # FIX 3: pre-allocated batch tensors
    batch_buf = BatchBuffer(BATCH_SIZE)

    agent = DQNAgent(player_id=1)
    agent.policy_net = policy_net
    agent.epsilon    = EPS_START

    # Curriculum
    curriculum_stages = ["Random", "RuleBased", "Self"]
    stage_idx  = 0
    stage_name = curriculum_stages[stage_idx]
    recent_outcomes = collections.deque(maxlen=500)

    def make_opponent(name):
        if name == "Random":
            return RandomAgent(player_id=2)
        elif name == "RuleBased":
            return RuleBasedAgent(player_id=2)
        else:
            snap = DQNAgent(player_id=2)
            snap.policy_net.load_state_dict(policy_net.state_dict())
            snap.epsilon = 0.05
            return snap

    opponent = make_opponent(stage_name)

    step_counter  = [0]
    best_win_rate = 0.0
    total_loss    = 0.0
    loss_count    = 0
    t0            = time.time()

    print(f"\nStarting training — {EPISODES} episodes")
    print(f"Stage 0: vs {stage_name}\n")

    for episode in range(1, EPISODES + 1):

        # Linear epsilon decay
        agent.epsilon = max(
            EPS_END,
            EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_STEPS)
        )

        # FIX 1: anneal PER beta toward 1.0
        beta = min(1.0,
                   PER_BETA_START + (1.0 - PER_BETA_START)
                   * (step_counter[0] / PER_BETA_STEPS))

        outcome = run_episode(agent, opponent, replay, device, step_counter)
        recent_outcomes.append(outcome)

        # Learn
        if step_counter[0] % TRAIN_EVERY == 0:
            loss = learn(policy_net, target_net, optimizer,
                         replay, batch_buf, device, beta)
            if loss is not None:
                total_loss += loss
                loss_count += 1

        # FIX 2: step LR scheduler every episode
        scheduler.step(episode)

        # Sync target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Curriculum check
        if len(recent_outcomes) == 500:
            win_rate = recent_outcomes.count("win") / 500
            if win_rate >= CURRICULUM_THRESHOLD and stage_idx < len(curriculum_stages) - 1:
                stage_idx += 1
                stage_name = curriculum_stages[stage_idx]
                opponent   = make_opponent(stage_name)
                recent_outcomes.clear()
                print(f"\n  >>> Graduated to stage {stage_idx}: vs {stage_name}"
                      f"  (win-rate was {win_rate:.0%})\n")

            if stage_name == "Self" and episode % 2_000 == 0:
                opponent = make_opponent("Self")

        # Periodic checkpoint
        if episode % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"dqn_ep{episode}.pth")
            torch.save(policy_net.state_dict(), ckpt_path)

        # Evaluation
        if episode % EVAL_EVERY == 0:
            wr_dict  = evaluate(agent, device)   # FIX 6
            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            elapsed  = time.time() - t0
            cur_lr   = scheduler.get_last_lr()[0]
            print(
                f"Ep {episode:>6} | stage={stage_name:<9} | "
                f"wr_rnd={wr_dict['random']:.0%} "
                f"wr_rule={wr_dict['rule']:.0%} | "
                f"eps={agent.epsilon:.3f} | "
                f"loss={avg_loss:.4f} | "
                f"lr={cur_lr:.2e} | "
                f"buf={len(replay):>6} | t={elapsed:.0f}s"
            )
            total_loss = 0.0
            loss_count = 0

            combined_wr = (wr_dict["random"] + wr_dict["rule"]) / 2
            if combined_wr > best_win_rate:
                best_win_rate = combined_wr
                best_path = "dqn_connect4_best.pth"
                torch.save(policy_net.state_dict(), best_path)
                print(f"  ★ New best model saved (combined {combined_wr:.0%})"
                      f" → {best_path}")

    final_path = os.path.join(SAVE_DIR, "dqn_final.pth")
    torch.save(policy_net.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best combined win-rate: {best_win_rate:.0%}")


if __name__ == "__main__":
    train()