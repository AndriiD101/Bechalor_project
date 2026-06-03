"""
Curriculum DQN — Connect-4
──────────────────────────
Improvements over v1
  • Performance-gated phase advance (win-rate threshold) instead of fixed episode counts
  • Prioritized Experience Replay (PER) with importance-sampling correction
  • Reward shaping: small bonuses for creating/blocking threats mid-game
  • Self-play final phase against a frozen snapshot of the best model
  • LR cosine-annealing scheduler across full training
  • Epsilon soft-reset (+0.15 bump, clamped to eps_start) on each phase advance
  • Single unified checkpoint file: weights + full metrics history in one .pt
  • MetricsTracker exposes a smoothed win-rate trend for phase-gate decisions
"""

import argparse
import collections
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.connect4 import Connect4Game
from agents.reinforcement_agent import Connect4Net
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.alfabetapruning_agent import AlphaBetaAgent


# ─── CURRICULUM DEFINITION ──────────────────────────────────────────────
# Each phase runs until BOTH conditions are met:
#   (a) at least `min_episodes` have been played in this phase
#   (b) smoothed eval win-rate >= `advance_threshold`
# If the agent never clears the threshold, it runs for at most `max_episodes`.
CURRICULUM = [
    {
        "type": "random",
        "config": {},
        "min_episodes":       1_000,
        "max_episodes":       4_000,
        "advance_threshold":  0.90,   # must win ≥90 % vs random
    },
    {
        "type": "rule_based",
        "config": {},
        "min_episodes":       3_000,
        "max_episodes":       10_000,
        "advance_threshold":  0.70,
    },
    {
        "type": "alphabeta",
        "config": {"max_depth": 2},
        "min_episodes":       5_000,
        "max_episodes":       15_000,
        "advance_threshold":  0.60,
    },
    {
        "type": "self_play",
        "config": {},
        "min_episodes":       5_000,
        "max_episodes":       10_000,
        "advance_threshold":  1.0,    # always runs to max_episodes
    },
]

DEFAULTS = dict(
    batch_size=256,
    replay_capacity=150_000,
    lr=3e-4,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.999_95,
    eps_phase_bump=0.15,       # epsilon boost on each phase advance
    target_sync=500,
    eval_every=500,
    eval_games=200,
    eval_eps=0.02,
    min_replay=5_000,
    per_alpha=0.6,             # PER: priority exponent
    per_beta_start=0.4,        # PER: IS correction start
    per_beta_end=1.0,          # PER: IS correction end
    reward_threat=0.05,        # shaping bonus per threat created/blocked
    smoothing_window=3,        # eval checkpoints to smooth over for gate
    save_dir="checkpoints",
    resume=None,
    checkpoint_name="dqn_curriculum.pt",
)


# ─── PRIORITIZED REPLAY BUFFER ───────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)

class SumTree:
    """Binary sum-tree for O(log n) priority updates and sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        # Clamp s so floating-point overshoot never walks past the last leaf
        s = min(s, self.tree[0] - 1e-6)
        idx = self._retrieve(0, s)
        # Ensure idx is within bounds of the tree array
        idx = min(idx, len(self.tree) - 1)
        data_idx = min(idx - self.capacity + 1, self.size - 1)
        data_idx = max(data_idx, 0)
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """Rank-based PER with importance-sampling weights."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.eps_priority = 1e-6

    def push(self, *args):
        self.tree.add(self.max_priority ** self.alpha, Transition(*args))

    def sample(self, n: int, beta: float):
        indices, priorities, transitions = [], [], []
        segment = self.tree.total / n
        for i in range(n):
            lo, hi = segment * i, segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, priority, t = self.tree.get(s)
            if t is None:          # guard against uninitialised slots
                continue
            indices.append(idx)
            priorities.append(priority)
            transitions.append(t)

        if not transitions:
            return None

        probs = np.array(priorities, dtype=np.float64) / (self.tree.total + 1e-9)
        probs = np.clip(probs, 1e-9, None)  # Prevent zero probabilities
        weights = (len(self.tree) * probs) ** (-beta)
        weights /= weights.max()

        s, a, r, ns, d = zip(*transitions)
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(ns),
            torch.tensor(d, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices: list, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            p = (abs(err) + self.eps_priority) ** self.alpha
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, p)

    def __len__(self) -> int:
        return len(self.tree)


# ─── STATE ENCODING ──────────────────────────────────────────────────────
def board_to_tensor(board, player_id: int, valid_moves: list, device) -> torch.Tensor:
    board_f = np.flipud(board)
    opp_id = 2 if player_id == 1 else 1
    valid_layer = np.zeros((6, 7), dtype=np.float32)
    for col in valid_moves:
        valid_layer[:, col] = 1.0
    state = np.stack([
        (board_f == player_id).astype(np.float32),
        (board_f == opp_id).astype(np.float32),
        valid_layer,
    ], axis=0)
    return torch.tensor(state, dtype=torch.float32).to(device)


def pick_action(q_values: torch.Tensor, valid_moves: list) -> int:
    masked = np.full(7, -1e9, dtype=np.float32)
    q_np = q_values.cpu().numpy()
    for col in valid_moves:
        masked[col] = q_np[col]
    return int(np.argmax(masked))


# ─── REWARD SHAPING ──────────────────────────────────────────────────────
def count_threats(board, player_id: int) -> int:
    """Count open three-in-a-row threats for player_id."""
    rows, cols = board.shape
    opp = 2 if player_id == 1 else 1
    threats = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                cells = []
                for k in range(4):
                    nr, nc = r + dr * k, c + dc * k
                    if 0 <= nr < rows and 0 <= nc < cols:
                        cells.append(board[nr, nc])
                    else:
                        break
                if len(cells) == 4 and cells.count(opp) == 0 and cells.count(player_id) == 3:
                    threats += 1
    return threats


def shaping_reward(board_before, board_after, player_id: int, shaping_scale: float) -> float:
    """Delta in threats, scaled. Positive = created new threat; negative = allowed one."""
    if shaping_scale == 0.0:
        return 0.0
    opp = 2 if player_id == 1 else 1
    my_delta  = count_threats(board_after, player_id) - count_threats(board_before, player_id)
    opp_delta = count_threats(board_after, opp)       - count_threats(board_before, opp)
    return shaping_scale * (my_delta - opp_delta)


# ─── OPPONENT FACTORY ────────────────────────────────────────────────────
def make_opponent(phase_type: str, phase_config: dict, opp_pid: int,
                  self_play_net=None, device=None):
    if phase_type == "random":
        return RandomAgent(player_id=opp_pid)
    if phase_type == "rule_based":
        return RuleBasedAgent(player_id=opp_pid)
    if phase_type == "alphabeta":
        return AlphaBetaAgent(player_id=opp_pid, max_depth=phase_config.get("max_depth", 2))
    if phase_type == "self_play":
        # Returns a callable that mimics agent.select_move interface
        assert self_play_net is not None and device is not None
        return _SelfPlayOpponent(self_play_net, opp_pid, device)
    raise ValueError(f"Unknown opponent type: {phase_type!r}")


class _SelfPlayOpponent:
    """Wraps a frozen policy net so it can be used like a rule-based agent."""

    def __init__(self, net: nn.Module, player_id: int, device):
        self.net = net
        self.player_id = player_id
        self.device = device

    def select_move(self, game) -> int:
        valid = game.get_valid_locations()
        if not valid:
            return 0
        state_t = board_to_tensor(game.board, self.player_id, valid, self.device)
        with torch.no_grad():
            q_vals = self.net(state_t.unsqueeze(0)).squeeze(0)
        return pick_action(q_vals, valid)


# ─── EVALUATION ──────────────────────────────────────────────────────────
def evaluate(policy_net: nn.Module, phase_type: str, phase_config: dict,
             n_games: int, device, eval_eps: float,
             self_play_net=None) -> dict:
    wins = draws = losses = 0

    for swap in (False, True):
        policy_pid = 2 if swap else 1
        opp_pid    = 1 if swap else 2

        for _ in range(n_games // 2):
            opponent = make_opponent(phase_type, phase_config, opp_pid,
                                     self_play_net=self_play_net, device=device)
            game  = Connect4Game()
            done  = False

            while not done:
                current_pid = game.current_player
                valid = game.get_valid_locations()

                if not valid:
                    draws += 1
                    break

                if current_pid == policy_pid:
                    if random.random() < eval_eps:
                        col = random.choice(valid)
                    else:
                        state_t = board_to_tensor(game.board, policy_pid, valid, device)
                        with torch.no_grad():
                            q_vals = policy_net(state_t.unsqueeze(0)).squeeze(0)
                        col = pick_action(q_vals, valid)
                else:
                    col = opponent.select_move(game)

                _, row = game.make_move(col)
                if row is not None and game.check_winner(row, col, current_pid):
                    (wins if current_pid == policy_pid else losses).__class__  # just a type hint
                    if current_pid == policy_pid:
                        wins += 1
                    else:
                        losses += 1
                    done = True
                elif game.check_draw():
                    draws += 1
                    done = True
                else:
                    game.switch_player()

    total = wins + draws + losses
    return {
        "win_rate":  wins   / total if total else 0.0,
        "draw_rate": draws  / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
    }


# ─── METRICS TRACKER ─────────────────────────────────────────────────────
class MetricsTracker:
    def __init__(self, smoothing_window: int = 3):
        self.window = smoothing_window
        self.history: list[dict] = []

    def add(self, episode: int, train_wr: float, phase: str, eval_stats: dict,
            epsilon: float, phase_idx: int, composite: float):
        self.history.append({
            "episode":   episode,
            "phase_idx": phase_idx,
            "phase":     phase,
            "epsilon":   round(epsilon, 5),
            "train_wr":  round(train_wr, 4),
            "eval_wr":   round(eval_stats["win_rate"],  4),
            "eval_dr":   round(eval_stats["draw_rate"], 4),
            "eval_lr":   round(eval_stats["loss_rate"], 4),
            "composite": round(composite, 4),
        })

    def smoothed_eval_wr(self) -> float:
        """Average eval win-rate over the last `window` checkpoints."""
        recent = [h["eval_wr"] for h in self.history[-self.window:]]
        return sum(recent) / len(recent) if recent else 0.0

    def phase_history(self, phase_idx: int) -> list[dict]:
        return [h for h in self.history if h["phase_idx"] == phase_idx]

    def to_dict(self) -> dict:
        return {"smoothing_window": self.window, "records": self.history}


# ─── CHECKPOINT MANAGER ──────────────────────────────────────────────────
class CheckpointManager:
    """
    Saves a single .pt file containing both the model weights and the full
    metrics history, so nothing is ever split across files.
    """

    def __init__(self, save_dir: Path, checkpoint_name: str):
        self.save_dir = save_dir
        self.checkpoint_name = checkpoint_name
        self.path = save_dir / checkpoint_name
        self.best_score = -float("inf")
        self.best_episode = 0

    @staticmethod
    def composite_score(eval_stats: dict, train_wr: float) -> float:
        wr = eval_stats.get("win_rate",  0.0)
        dr = eval_stats.get("draw_rate", 0.0)
        lr = eval_stats.get("loss_rate", 0.0)
        return float(0.45 * wr + 0.15 * dr + 0.20 * train_wr - 0.20 * lr)

    def save(self, episode: int, score: float,
             policy_net: nn.Module, optimizer: optim.Optimizer,
             epsilon: float, metrics: MetricsTracker,
             phase_idx: int, cfg: dict) -> bool:
        """Always overwrites. Returns True if this is a new best score."""
        is_best = score > self.best_score
        if is_best:
            self.best_score   = score
            self.best_episode = episode

        payload = {
            # ── model state ──
            "model_state":     policy_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            # ── training state ──
            "episode":         episode,
            "epsilon":         epsilon,
            "phase_idx":       phase_idx,
            # ── scores ──
            "best_score":      self.best_score,
            "best_episode":    self.best_episode,
            "last_score":      score,
            # ── full history (human-readable too) ──
            "metrics":         metrics.to_dict(),
            # ── config snapshot ──
            "config":          cfg,
        }
        torch.save(payload, self.path)
        return is_best

    def load(self, policy_net: nn.Module, optimizer: optim.Optimizer,
             device, metrics: MetricsTracker):
        """Load from the unified checkpoint. Returns (episode, epsilon, phase_idx)."""
        payload = torch.load(self.path, map_location=device)
        policy_net.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        metrics.history = payload["metrics"]["records"]
        self.best_score   = payload["best_score"]
        self.best_episode = payload["best_episode"]
        return payload["episode"], payload["epsilon"], payload["phase_idx"]


# ─── TRAINING LOOP ───────────────────────────────────────────────────────
def train(cfg: dict):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir  = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Networks ──
    policy_net = Connect4Net().to(device)
    target_net = copy.deepcopy(policy_net).to(device)
    target_net.eval()

    optimizer  = optim.Adam(policy_net.parameters(), lr=cfg["lr"])

    # ── PER buffer ──
    replay = PrioritizedReplayBuffer(cfg["replay_capacity"], alpha=cfg["per_alpha"])

    # ── Checkpoint & metrics ──
    ckpt_mgr = CheckpointManager(save_dir, cfg["checkpoint_name"])
    metrics  = MetricsTracker(smoothing_window=cfg["smoothing_window"])

    epsilon      = cfg["eps_start"]
    global_step  = 0
    phase_idx    = 0
    phase_ep     = 0                  # episodes played within current phase
    ep_outcomes  = []
    self_play_net = None              # frozen opponent for self-play phase

    # ── Optional resume ──
    if cfg["resume"]:
        resume_path = Path(cfg["resume"])
        if resume_path.exists():
            episode_start, epsilon, phase_idx = ckpt_mgr.load(
                policy_net, optimizer, device, metrics
            )
            target_net.load_state_dict(policy_net.state_dict())
            print(f"[resume] Loaded from {resume_path}  (ep={episode_start}, phase={phase_idx})")
        else:
            print(f"[resume] Path not found: {resume_path} — starting fresh.")

    # ── LR scheduler: cosine anneal over total estimated episodes ──
    total_episodes = sum(p["max_episodes"] for p in CURRICULUM)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_episodes, eta_min=cfg["lr"] * 0.01
    )

    def current_phase_def() -> dict:
        return CURRICULUM[phase_idx]

    def build_self_play_snapshot():
        nonlocal self_play_net
        snap = copy.deepcopy(policy_net).to(device)
        snap.eval()
        self_play_net = snap
        print("  [self-play] Snapshot frozen.")

    # ── Phase transition helper ──
    def advance_phase():
        nonlocal phase_idx, phase_ep, epsilon, self_play_net
        phase_idx += 1
        phase_ep   = 0
        ckpt_mgr.best_score = -float("inf")   # reset gate for new phase

        # Soft epsilon reset — bump up so agent re-explores vs new opponent
        epsilon = min(cfg["eps_start"], epsilon + cfg["eps_phase_bump"])

        p = current_phase_def()
        if p["type"] == "self_play":
            build_self_play_snapshot()

        print(f"\n{'='*55}")
        print(f"  PHASE {phase_idx + 1}: vs {p['type'].upper()}  {p['config']}")
        print(f"  epsilon reset to {epsilon:.3f}")
        print(f"{'='*55}\n")

    # ── Print header ──
    print(f"\nDevice: {device}")
    print(f"Total max episodes: {total_episodes:,}")
    print(f"Checkpoint: {ckpt_mgr.path}\n")
    print(f"Phase 1: vs {CURRICULUM[0]['type'].upper()}")

    episode = 0

    while phase_idx < len(CURRICULUM):
        phase_def = current_phase_def()
        episode  += 1
        phase_ep += 1

        # ── Sample beta for PER IS correction (anneal toward 1.0) ──
        frac   = min(1.0, episode / total_episodes)
        beta   = cfg["per_beta_start"] + frac * (cfg["per_beta_end"] - cfg["per_beta_start"])

        # ── Game setup ──
        dqn_pid = random.choice([1, 2])
        opp_pid = 3 - dqn_pid

        opponent = make_opponent(
            phase_def["type"], phase_def["config"], opp_pid,
            self_play_net=self_play_net, device=device
        )

        game      = Connect4Game()
        ep_reward = 0.0
        done      = False
        last_state, last_action, last_board = None, None, None

        while not done:
            pid   = game.current_player
            valid = game.get_valid_locations()

            if not valid:
                if last_state is not None:
                    replay.push(last_state, last_action, 0.2,
                                board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                    ep_reward += 0.2
                break

            if pid == dqn_pid:
                board_before = game.board.copy()
                state_t = board_to_tensor(game.board, dqn_pid, valid, device)

                if random.random() < epsilon:
                    col = random.choice(valid)
                else:
                    with torch.no_grad():
                        col = pick_action(policy_net(state_t.unsqueeze(0)).squeeze(0), valid)

                _, row = game.make_move(col)
                board_after = game.board.copy()

                # Intermediate shaping reward
                shape = shaping_reward(board_before, board_after, dqn_pid, cfg["reward_threat"])

                if row is not None and game.check_winner(row, col, dqn_pid):
                    r = 1.0 + shape
                    replay.push(state_t, col, r,
                                board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                    ep_reward += r
                    done = True
                elif game.check_draw():
                    r = 0.3 + shape
                    replay.push(state_t, col, r,
                                board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                    ep_reward += r
                    done = True
                else:
                    last_state, last_action, last_board = state_t, col, board_after
                    game.switch_player()

            else:
                board_before = game.board.copy()
                col = opponent.select_move(game)
                _, row = game.make_move(col)
                board_after = game.board.copy()

                if row is not None and game.check_winner(row, col, opp_pid):
                    r = -1.0
                    if last_state is not None:
                        replay.push(last_state, last_action, r,
                                    board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                        ep_reward += r
                    done = True
                elif game.check_draw():
                    r = 0.3
                    if last_state is not None:
                        replay.push(last_state, last_action, r,
                                    board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                        ep_reward += r
                    done = True
                else:
                    # Step transition for the DQN's previous move
                    if last_state is not None:
                        # Small shaping: opponent created a threat = bad for us
                        opp_shape = shaping_reward(
                            board_before, board_after, opp_pid, cfg["reward_threat"]
                        )
                        next_valid = game.get_valid_locations()
                        replay.push(last_state, last_action, -opp_shape,
                                    board_to_tensor(game.board, dqn_pid, next_valid, device), 0.0)
                        last_state = last_action = last_board = None
                    game.switch_player()

            # ── Training Step ──
            if len(replay) >= max(cfg["min_replay"], cfg["batch_size"]):
                batch = replay.sample(cfg["batch_size"], beta)
                if batch is not None:
                    states, actions, rewards, next_states, dones, weights, tree_idxs = batch
                    states, actions, rewards = states.to(device), actions.to(device), rewards.to(device)
                    next_states, dones, weights = next_states.to(device), dones.to(device), weights.to(device)

                    q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_acts  = policy_net(next_states).argmax(1, keepdim=True)
                        target_q   = rewards + cfg["gamma"] * (
                            target_net(next_states).gather(1, next_acts).squeeze(1) * (1.0 - dones)
                        )

                    td_errors = (q_vals - target_q).detach().cpu().numpy()
                    replay.update_priorities(tree_idxs, td_errors)

                    loss = (weights * nn.SmoothL1Loss(reduction="none")(q_vals, target_q)).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                    optimizer.step()
                    scheduler.step()

                    if global_step % cfg["target_sync"] == 0:
                        target_net.load_state_dict(policy_net.state_dict())

            epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])
            global_step += 1

        ep_outcomes.append(1 if ep_reward > 0.5 else (0 if ep_reward < -0.5 else -1))

        # ── Logging & Evaluation ──
        if episode % cfg["eval_every"] == 0:
            recent = ep_outcomes[-cfg["eval_every"]:]
            w, l  = recent.count(1), recent.count(0)
            train_wr = w / (w + l) if (w + l) > 0 else 0.0

            policy_net.eval()
            eval_stats = evaluate(
                policy_net,
                phase_type=phase_def["type"],
                phase_config=phase_def["config"],
                n_games=cfg["eval_games"],
                device=device,
                eval_eps=cfg["eval_eps"],
                self_play_net=self_play_net,
            )
            policy_net.train()

            score    = CheckpointManager.composite_score(eval_stats, train_wr)
            is_best  = ckpt_mgr.save(
                episode, score, policy_net, optimizer,
                epsilon, metrics, phase_idx, cfg
            )
            metrics.add(episode, train_wr, phase_def["type"], eval_stats,
                        epsilon, phase_idx, score)

            lr_now  = scheduler.get_last_lr()[0]
            marker  = "  ★ best" if is_best else ""
            smooth_wr = metrics.smoothed_eval_wr()

            print(
                f"[ep {episode:>7,}|ph {phase_idx+1}|ep_ph {phase_ep:>5,}]  "
                f"vs {phase_def['type'][:8]:<8}  "
                f"train={train_wr:.3f}  "
                f"eval={eval_stats['win_rate']:.3f}(↑{smooth_wr:.3f})  "
                f"dr={eval_stats['draw_rate']:.3f}  lr={eval_stats['loss_rate']:.3f}  "
                f"score={score:.3f}  ε={epsilon:.4f}  lr={lr_now:.2e}"
                f"{marker}"
            )

            # ── Phase-advance gate ──
            can_advance = phase_ep >= phase_def["min_episodes"]
            threshold_met = smooth_wr >= phase_def["advance_threshold"]
            timed_out     = phase_ep >= phase_def["max_episodes"]

            if can_advance and (threshold_met or timed_out):
                reason = "threshold" if threshold_met else "max_episodes"
                print(f"\n  [phase gate] Advancing — reason: {reason}  "
                      f"smooth_wr={smooth_wr:.3f} >= {phase_def['advance_threshold']}")
                if phase_idx + 1 < len(CURRICULUM):
                    advance_phase()
                else:
                    print("\n  All phases complete.")
                    break

    # ── Final save ──
    ckpt_mgr.save(episode, ckpt_mgr.best_score, policy_net, optimizer,
                  epsilon, metrics, phase_idx, cfg)

    print(f"\n{'='*65}")
    print(f"Training complete.  Best episode: {ckpt_mgr.best_episode:,}")
    print(f"Best composite score: {ckpt_mgr.best_score:.4f}")
    print(f"Checkpoint: {ckpt_mgr.path}")
    print(f"  Contains: model weights + optimizer state + full metrics history")
    print(f"{'='*65}")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────
def parse_args() -> dict:
    p = argparse.ArgumentParser(description="Curriculum DQN for Connect-4")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", default=v, type=str if v is None else type(v))
    return vars(p.parse_args())


if __name__ == "__main__":
    train(parse_args())