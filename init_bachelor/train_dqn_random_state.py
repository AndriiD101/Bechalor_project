import argparse
import collections
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game.connect4_env import Connect4Env
from game.connect4 import Connect4Game
from agents.reinforcement_agent import Connect4Net

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    episodes=30_000,
    batch_size=256,
    replay_capacity=120_000,
    lr=1e-4,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.999_95,
    eps_warmup=5_000,
    train_freq=4,              # Opt: Backprop every N env steps to balance collection/compute
    tau=0.005,                 # Opt: Soft update for target network (set to 1.0 for hard updates)
    target_sync=500,           # Only used if tau == 1.0
    eval_every=1_000,
    eval_games=200,
    min_replay=5_000,
    save_dir="checkpoints",
    resume=None,
    # PER
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_frames=100_000,
    # N-step
    n_steps=3,
    # NoisyNet
    noisy_sigma0=0.5,
    use_noisy=True,
    # Random-state curriculum
    rand_moves_start=20,
    rand_moves_end=4,
    rand_anneal_episodes=15_000,
    max_rand_attempts=50,
)

Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)

# ---------------------------------------------------------------------------
# NoisyLinear
# ---------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet (Fortunato et al., 2017)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), sigma_init / math.sqrt(in_features))
        )
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(
            torch.full((out_features,), sigma_init / math.sqrt(out_features))
        )
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_i = self._f(torch.randn(self.in_features))
        eps_j = self._f(torch.randn(self.out_features))
        self.weight_epsilon.copy_(eps_j.outer(eps_i))
        self.bias_epsilon.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyConnect4Net(Connect4Net):
    """Connect4Net with the last two Linear layers replaced by NoisyLinear."""
    def __init__(self, sigma_init: float = 0.5):
        super().__init__()
        linear_layers = [
            (name, mod) for name, mod in self.named_modules()
            if isinstance(mod, nn.Linear)
        ]
        if len(linear_layers) < 2:
            raise RuntimeError("Base network must have at least 2 Linear layers.")
        for name, mod in linear_layers[-2:]:
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = self if not parent_name else dict(self.named_modules())[parent_name]
            setattr(parent, attr, NoisyLinear(mod.in_features, mod.out_features, sigma_init))

    def sample_noise(self):
        for mod in self.modules():
            if isinstance(mod, NoisyLinear):
                mod.sample_noise()


# ---------------------------------------------------------------------------
# Prioritized Experience Replay
# ---------------------------------------------------------------------------
class SumTree:
    # (Remains unchanged from your original script)
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write    = 0
        self.size     = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size  = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        first_leaf      = self.capacity - 1
        last_valid_leaf = first_leaf + self.size - 1
        idx      = max(first_leaf, min(idx, last_valid_leaf))
        data_idx = idx - first_leaf
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-5):
        self.tree         = SumTree(capacity)
        self.alpha        = alpha
        self.epsilon      = epsilon
        self.max_priority = 1.0

    def push(self, *args):
        self.tree.add(self.max_priority ** self.alpha, Transition(*args))

    def sample(self, n: int, beta: float = 0.4):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total / n
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is None:
                s = random.uniform(0, self.tree.total)
                idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        probs   = np.array(priorities, dtype=np.float64) / self.tree.total
        probs   = np.clip(probs, 1e-10, None)
        weights = (len(self.tree) * probs) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        s, a, r, ns, d = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(ns),
            torch.tensor(d, dtype=torch.float32),
            idxs,
            weights,
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            priority          = (abs(float(err)) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)


# ---------------------------------------------------------------------------
# N-step Return Buffer
# ---------------------------------------------------------------------------
class NStepBuffer:
    # (Remains unchanged)
    def __init__(self, n: int, gamma: float):
        self.n     = n
        self.gamma = gamma
        self.buf: collections.deque = collections.deque()

    def push(self, state, action, reward, next_state, done):
        self.buf.append(Transition(state, action, reward, next_state, done))
        if len(self.buf) < self.n and not done:
            return None
        R = 0.0
        for i, t in enumerate(self.buf):
            R += (self.gamma ** i) * t.reward
            if t.done:
                break
        first = self.buf[0]
        last  = self.buf[-1]
        if done:
            self.buf.clear()
        else:
            self.buf.popleft()
        return Transition(first.state, first.action, R, last.next_state, last.done)

    def flush(self):
        results = []
        while self.buf:
            R = sum((self.gamma ** i) * t.reward for i, t in enumerate(self.buf))
            first, last = self.buf[0], self.buf[-1]
            results.append(Transition(first.state, first.action, R, last.next_state, last.done))
            self.buf.popleft()
        return results


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def obs_to_tensor(obs: np.ndarray, player_id: int, valid_moves: list, device) -> torch.Tensor:
    """Optimized: Converts observation to tensor directly on target device."""
    board_t = torch.from_numpy(np.flipud(obs).copy()).to(device, dtype=torch.float32)
    opp_id = 2.0 if player_id == 1 else 1.0

    p1_layer = (board_t == float(player_id)).float()
    p2_layer = (board_t == opp_id).float()
    
    valid_layer = torch.zeros((6, 7), dtype=torch.float32, device=device)
    if valid_moves:
        valid_layer[:, valid_moves] = 1.0

    return torch.stack([p1_layer, p2_layer, valid_layer], dim=0)


def get_valid_moves(env: Connect4Env) -> list[int]:
    game = env.game
    return [c for c in range(7) if game.is_valid_location(game.board, c)]


def pick_action(q_values: torch.Tensor, valid_moves: list) -> int:
    masked = np.full(7, -1e9, dtype=np.float32)
    q_np   = q_values.cpu().numpy()
    for col in valid_moves:
        masked[col] = q_np[col]
    return int(np.argmax(masked))


def beta_schedule(frame: int, beta_start: float, beta_frames: int) -> float:
    return min(1.0, beta_start + frame * (1.0 - beta_start) / beta_frames)


# ---------------------------------------------------------------------------
# Checkpoint & Random-state curriculum (Truncated for brevity, un-changed)
# ---------------------------------------------------------------------------
def _is_terminal(game: Connect4Game) -> bool:
    if game.check_draw(): return True
    for piece in (1, 2):
        if game.winning_move(piece): return True
    return False

def make_random_start_env(rand_moves: int, max_attempts: int = 50) -> Connect4Env:
    for _ in range(max_attempts):
        game = Connect4Game()
        moves_played = 0
        for _ in range(rand_moves):
            valid = game.get_valid_locations()
            if not valid: break
            col = random.choice(valid)
            _, row = game.make_move(col)
            moves_played += 1
            if row is not None and game.check_winner(row, col, game.current_player):
                break
            game.switch_player()
        if _is_terminal(game): continue

        env = Connect4Env(opponent_agent=None, render_mode=None)
        env._game = game
        return env
    
    env = Connect4Env(opponent_agent=None, render_mode=None)
    env._game = Connect4Game()
    return env

def current_rand_moves(episode: int, cfg: dict) -> int:
    start = cfg["rand_moves_start"]
    end   = cfg["rand_moves_end"]
    if start == end or episode >= cfg["rand_anneal_episodes"]:
        return end
    frac = episode / cfg["rand_anneal_episodes"]
    return round(start + frac * (end - start))

class CheckpointManager:
    # (Remains unchanged)
    def __init__(self, save_dir: Path, model_name: str = "best_model.pt"):
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_score = -float("inf")
        self.best_step = 0
        self.best_path = None

    def compute_composite_score(self, stats: dict, train_wr: float) -> float:
        eval_wr = stats["win_rate"]
        eval_dr = stats["draw_rate"]
        eval_lr = stats["loss_rate"]
        score = (0.4 * eval_wr + 0.2 * eval_dr + 0.2 * train_wr + -0.2 * eval_lr)
        return float(score)

    def update_best(self, step: int, score: float, state_dict) -> bool:
        if score > self.best_score:
            if self.best_path and self.best_path.exists():
                self.best_path.unlink()
            self.best_score = score
            self.best_step = step
            self.best_path = self.save_dir / self.model_name
            torch.save(state_dict, self.best_path)
            return True
        return False

    def get_best_checkpoint(self) -> dict | None:
        if self.best_path is None: return None
        return {"step": self.best_step, "score": self.best_score, "path": self.best_path}

class MetricsTracker:
    def __init__(self, window: int = 5):
        self.window = window
        self.history: list[dict] = []

    def add(self, step: int, eval_stats: dict, train_wr: float):
        self.history.append({
            "step": step, "eval_wr": eval_stats["win_rate"],
            "eval_dr": eval_stats["draw_rate"], "eval_lr": eval_stats["loss_rate"],
            "train_wr": train_wr,
        })

    def get_ema(self, metric: str, alpha: float = 0.3) -> float:
        if not self.history: return 0.0
        ema = self.history[0][metric]
        for entry in self.history[1:]:
            ema = alpha * entry[metric] + (1 - alpha) * ema
        return ema

    def get_smoothed_stats(self, alpha: float = 0.3) -> dict:
        return {
            "eval_wr_ema": self.get_ema("eval_wr", alpha),
            "eval_dr_ema": self.get_ema("eval_dr", alpha),
            "eval_lr_ema": self.get_ema("eval_lr", alpha),
            "train_wr_ema": self.get_ema("train_wr", alpha),
        }

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(policy_net, snapshot_net, n_games: int, device) -> dict:
    wins = draws = losses = 0

    for swap in (False, True):
        for _ in range(n_games // 2):
            env = Connect4Env(render_mode=None)
            obs, info = env.reset()
            done      = False
            player_id = 1

            while not done:
                valid = get_valid_moves(env)
                if not valid:
                    draws += 1
                    break

                net     = policy_net if (player_id == 1) ^ swap else snapshot_net
                state_t = obs_to_tensor(obs, player_id, valid, device)
                with torch.no_grad():
                    q_vals = net(state_t.unsqueeze(0)).squeeze(0)
                action = pick_action(q_vals, valid)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    if reward > 0:
                        if (player_id == 1) ^ swap: wins += 1
                        else: losses += 1
                    elif reward < 0:
                        if (player_id == 1) ^ swap: losses += 1
                        else: wins += 1
                    else:
                        draws += 1
                else:
                    player_id = 3 - player_id
            env.close()

    total = wins + draws + losses
    return {
        "win_rate":  wins   / total if total else 0.0,
        "draw_rate": draws  / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
    }

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(cfg: dict):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize AMP GradScaler
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg["use_noisy"]:
        policy_net = NoisyConnect4Net(sigma_init=cfg["noisy_sigma0"]).to(device)
    else:
        policy_net = Connect4Net().to(device)

    target_net   = copy.deepcopy(policy_net).to(device)
    target_net.eval()
    snapshot_net = copy.deepcopy(policy_net).to(device)
    snapshot_net.eval()

    if cfg["resume"]:
        policy_net.load_state_dict(torch.load(cfg["resume"], map_location=device))

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg["lr"])

    replay     = PrioritizedReplayBuffer(cfg["replay_capacity"], alpha=cfg["per_alpha"])
    nstep_bufs = {
        1: NStepBuffer(cfg["n_steps"], cfg["gamma"]),
        2: NStepBuffer(cfg["n_steps"], cfg["gamma"]),
    }

    checkpoint_mgr = CheckpointManager(save_dir)
    metrics_tracker = MetricsTracker(window=10)

    epsilon     = cfg["eps_start"]
    best_score  = -float("inf")
    global_step = 0
    ep_outcomes: list[int] = []

    for episode in range(1, cfg["episodes"] + 1):
        rand_moves = current_rand_moves(episode, cfg)
        env        = make_random_start_env(rand_moves, cfg["max_rand_attempts"])

        player_id   = env.game.current_player
        obs         = env._get_obs()
        done        = False
        last_state  = {1: None, 2: None}
        last_action = {1: None, 2: None}
        ep_reward   = 0.0

        # Sample noise at the start of the episode for exploration
        if cfg["use_noisy"] and hasattr(policy_net, "sample_noise"):
            policy_net.sample_noise()

        while not done:
            valid = get_valid_moves(env)
            if not valid:
                ep_reward = 0.05
                ep_outcomes.append(-1)
                break

            opp     = 3 - player_id
            state_t = obs_to_tensor(obs, player_id, valid, device)

            if last_state[opp] is not None:
                next_state_opp = obs_to_tensor(obs, opp, valid, device)
                t = nstep_bufs[opp].push(last_state[opp], last_action[opp], 0.0, next_state_opp, False)
                if t is not None:
                    replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

            use_epsilon = (not cfg["use_noisy"]) or (global_step < cfg["eps_warmup"])
            if use_epsilon and random.random() < epsilon:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    q_vals = policy_net(state_t.unsqueeze(0)).squeeze(0)
                action = pick_action(q_vals, valid)

            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done and reward > 0.5:
                term_pid = obs_to_tensor(obs_next, player_id, [], device)
                term_opp = obs_to_tensor(obs_next, opp,       [], device)

                t = nstep_bufs[player_id].push(state_t, action, 1.0, term_pid, True)
                if t: replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))
                for t in nstep_bufs[player_id].flush():
                    replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

                for _ in nstep_bufs[opp].flush(): pass
                if last_state[opp] is not None:
                    t = nstep_bufs[opp].push(last_state[opp], last_action[opp], -1.0, term_opp, True)
                    if t: replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))
                for t in nstep_bufs[opp].flush():
                    replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

                ep_reward = 1.0 if player_id == 1 else -1.0
                ep_outcomes.append(1 if player_id == 1 else 0)

            elif done:
                term_pid = obs_to_tensor(obs_next, player_id, [], device)
                term_opp = obs_to_tensor(obs_next, opp,       [], device)

                for p in (player_id, opp):
                    for _ in nstep_bufs[p].flush(): pass

                for p, s in ((player_id, term_pid), (opp, term_opp)):
                    if last_state[p] is not None:
                        t = nstep_bufs[p].push(last_state[p], last_action[p], 0.05, s, True)
                        if t: replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))
                    for t in nstep_bufs[p].flush():
                        replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

                t = nstep_bufs[player_id].push(state_t, action, 0.05, term_pid, True)
                if t: replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

                ep_reward = 0.05
                ep_outcomes.append(-1)

            else:
                last_state[player_id]  = state_t
                last_action[player_id] = action
                player_id = 3 - player_id

            # ── Training step (Now controlled by train_freq) ──────────── #
            if len(replay) >= max(cfg["min_replay"], cfg["batch_size"]) and global_step % cfg["train_freq"] == 0:
                beta = beta_schedule(global_step, cfg["per_beta_start"], cfg["per_beta_frames"])
                states, actions, rewards, next_states, dones, idxs, weights = replay.sample(cfg["batch_size"], beta)

                states      = states.to(device)
                actions     = actions.to(device)
                rewards     = rewards.to(device)
                next_states = next_states.to(device)
                dones       = dones.to(device)
                weights     = weights.to(device)

                # CRITICAL: Resample noise for the batch forward pass
                if cfg["use_noisy"]:
                    policy_net.sample_noise()
                    if hasattr(target_net, "sample_noise"):
                        target_net.sample_noise()

                # Execute forward pass with Mixed Precision
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(scaler is not None)):
                    q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_acts = policy_net(next_states).argmax(1, keepdim=True)
                        target_q  = rewards + (cfg["gamma"] ** cfg["n_steps"]) * \
                                    target_net(next_states).gather(1, next_acts).squeeze(1) * \
                                    (1.0 - dones)
                    
                    loss = (weights * F.smooth_l1_loss(q_vals, target_q, reduction="none")).mean()

                optimizer.zero_grad(set_to_none=True)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer) # Unscale before clipping
                    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                    optimizer.step()

                # Soft or Hard Target Update
                if cfg["tau"] < 1.0:
                    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                        target_param.data.copy_(cfg["tau"] * policy_param.data + (1.0 - cfg["tau"]) * target_param.data)
                elif global_step % cfg["target_sync"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            global_step += 1
            obs = obs_next

        env.close()

        # Epsilon decay
        if global_step >= cfg["eps_warmup"]:
            epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])

        # Periodic evaluation & checkpointing
        if episode % cfg["eval_every"] == 0:
            recent    = ep_outcomes[-cfg["eval_every"]:]
            wins_tr   = recent.count(1)
            losses_tr = recent.count(0)
            denom     = wins_tr + losses_tr
            train_wr  = wins_tr / denom if denom else 0.0

            policy_net.eval()
            stats = evaluate(policy_net, snapshot_net, cfg["eval_games"], device)
            policy_net.train()

            metrics_tracker.add(episode, stats, train_wr)
            composite_score = checkpoint_mgr.compute_composite_score(stats, train_wr)
            is_best = checkpoint_mgr.update_best(episode, composite_score, policy_net.state_dict())

            if is_best:
                snapshot_net.load_state_dict(policy_net.state_dict())
                marker = "  ⭐ NEW BEST"
            else:
                marker = ""

            rm_now = current_rand_moves(episode, cfg)

            print(
                f"[ep {episode:>7,}]  "
                f"rand_moves={rm_now:>2d}  "
                f"train_wr={train_wr:.3f}  "
                f"eval_wr={stats['win_rate']:.3f}  "
                f"eval_dr={stats['draw_rate']:.3f}  "
                f"eval_lr={stats['loss_rate']:.3f}  "
                f"composite_score={composite_score:.4f}  "
                f"eps={epsilon:.4f}  "
                f"replay={len(replay):,}"
                f"{marker}"
            )

        if episode % 1_000 == 0:
            periodic_path = save_dir / f"periodic_ep{episode:06d}.pt"
            torch.save(policy_net.state_dict(), periodic_path)

if __name__ == "__main__":
    train(parse_args())