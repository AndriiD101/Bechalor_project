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

from game.connect4_env import Connect4Env          # ← Gymnasium wrapper (unified)
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
    target_sync=500,
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
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init / math.sqrt(in_features)))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init / math.sqrt(out_features)))
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
    def __init__(self, sigma_init: float = 0.5):
        super().__init__()
        linear_layers = [(name, mod) for name, mod in self.named_modules() if isinstance(mod, nn.Linear)]
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
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree  = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write = 0
        self.size  = 0

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
        idx = max(first_leaf, min(idx, last_valid_leaf))
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
    def __init__(self, n: int, gamma: float):
        self.n     = n
        self.gamma = gamma
        self.buf: collections.deque = collections.deque()

    def push(self, state, action, reward, next_state, done):
        self.buf.append(Transition(state, action, reward, next_state, done))
        if len(self.buf) < self.n and not done:
            return None
        R     = 0.0
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
# Utility functions
# ---------------------------------------------------------------------------
def obs_to_tensor(obs: np.ndarray, player_id: int, valid_moves: list, device) -> torch.Tensor:
    """Convert Gymnasium observation (6×7) to 3-channel tensor [3, 6, 7]."""
    board_f   = np.flipud(obs.astype(np.float32))
    opp_id    = 2 if player_id == 1 else 1
    valid_layer = np.zeros((6, 7), dtype=np.float32)
    for col in valid_moves:
        valid_layer[:, col] = 1.0
    state = np.stack([
        (board_f == player_id).astype(np.float32),
        (board_f == opp_id).astype(np.float32),
        valid_layer,
    ], axis=0)
    return torch.tensor(state, dtype=torch.float32).to(device)


def get_valid_moves(env: Connect4Env) -> list[int]:
    """Get valid column indices from the live environment."""
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
# Evaluation — both agents go through Connect4Env
# ---------------------------------------------------------------------------
def evaluate(policy_net, snapshot_net, n_games: int, device) -> dict:
    """
    n_games/2 as P1=policy vs P2=snapshot, then n_games/2 swapped.
    All interaction via Connect4Env.step() / reset().
    """
    wins = draws = losses = 0

    for swap in (False, True):
        for _ in range(n_games // 2):
            env = Connect4Env(render_mode=None)
            obs, info = env.reset()
            done      = False
            player_id = 1  # P1 always moves first in env

            while not done:
                valid = get_valid_moves(env)
                if not valid:
                    draws += 1
                    break

                # policy_net plays as the non-swapped side
                net       = policy_net if (player_id == 1) ^ swap else snapshot_net
                state_t   = obs_to_tensor(obs, player_id, valid, device)
                with torch.no_grad():
                    q_vals = net(state_t.unsqueeze(0)).squeeze(0)
                action = pick_action(q_vals, valid)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    if reward > 0:          # acting player won
                        if (player_id == 1) ^ swap:
                            wins += 1
                        else:
                            losses += 1
                    elif reward < 0:        # acting player lost
                        if (player_id == 1) ^ swap:
                            losses += 1
                        else:
                            wins += 1
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
# Main training loop — per-player buffers via Connect4Env
# ---------------------------------------------------------------------------
def train(cfg: dict):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Networks
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
        print(f"[resume] Loaded weights from {cfg['resume']}")

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg["lr"])

    replay     = PrioritizedReplayBuffer(cfg["replay_capacity"], alpha=cfg["per_alpha"])
    nstep_bufs = {
        1: NStepBuffer(cfg["n_steps"], cfg["gamma"]),
        2: NStepBuffer(cfg["n_steps"], cfg["gamma"]),
    }

    epsilon     = cfg["eps_start"]
    best_wr     = -1.0
    global_step = 0
    ep_outcomes: list[int] = []   # 1 = P1 win, 0 = P2 win, -1 = draw

    for episode in range(1, cfg["episodes"] + 1):

        # ── Episode setup ────────────────────────────────────────────── #
        env           = Connect4Env(render_mode=None)
        obs, info     = env.reset()
        done          = False
        player_id     = 1  # env always starts with P1
        last_state    = {1: None, 2: None}
        last_action   = {1: None, 2: None}
        ep_reward     = 0.0

        if cfg["use_noisy"] and hasattr(policy_net, "sample_noise"):
            policy_net.sample_noise()

        while not done:
            valid = get_valid_moves(env)
            if not valid:          # full board without terminal — shouldn't happen
                ep_reward = 0.05
                ep_outcomes.append(-1)
                break

            opp     = 3 - player_id
            state_t = obs_to_tensor(obs, player_id, valid, device)

            # ── Action selection ─────────────────────────────────────── #
            use_epsilon = (not cfg["use_noisy"]) or (global_step < cfg["eps_warmup"])
            if use_epsilon and random.random() < epsilon:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    q_vals = policy_net(state_t.unsqueeze(0)).squeeze(0)
                action = pick_action(q_vals, valid)

            # ── Give opponent their non-terminal credit (pre-move) ───── #
            # The board right now is what the opponent "sees" as their
            # next_state (the state resulting from our previous move).
            if last_state[opp] is not None:
                next_state_opp = obs_to_tensor(obs, opp, valid, device)
                t = nstep_bufs[opp].push(last_state[opp], last_action[opp], 0.0, next_state_opp, False)
                if t is not None:
                    replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

            # ── Step the environment ─────────────────────────────────── #
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ── Terminal: current player wins (reward == +1) ─────────── #
            if done and reward > 0.5:
                term_pid = obs_to_tensor(obs_next, player_id, [], device)
                term_opp = obs_to_tensor(obs_next, opp,       [], device)

                t = nstep_bufs[player_id].push(state_t, action, 1.0, term_pid, True)
                if t:
                    replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))
                for t in nstep_bufs[player_id].flush():
                    replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

                # Loser — discard speculative non-terminal, push loss
                for _ in nstep_bufs[opp].flush():
                    pass
                if last_state[opp] is not None:
                    t = nstep_bufs[opp].push(last_state[opp], last_action[opp], -1.0, term_opp, True)
                    if t:
                        replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))
                for t in nstep_bufs[opp].flush():
                    replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

                ep_reward = 1.0 if player_id == 1 else -1.0
                ep_outcomes.append(1 if player_id == 1 else 0)

            # ── Terminal: illegal move (reward == -10) or draw (0) ───── #
            elif done:
                term_pid = obs_to_tensor(obs_next, player_id, [], device)
                term_opp = obs_to_tensor(obs_next, opp,       [], device)

                for p in (player_id, opp):
                    for _ in nstep_bufs[p].flush():
                        pass

                for p, s in ((player_id, term_pid), (opp, term_opp)):
                    if last_state[p] is not None:
                        t = nstep_bufs[p].push(last_state[p], last_action[p], 0.05, s, True)
                        if t:
                            replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))
                    for t in nstep_bufs[p].flush():
                        replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

                # Current player's draw move
                t = nstep_bufs[player_id].push(state_t, action, 0.05, term_pid, True)
                if t:
                    replay.push(t.state, t.action, t.reward, t.next_state, float(t.done))

                ep_reward = 0.05
                ep_outcomes.append(-1)

            # ── Non-terminal ─────────────────────────────────────────── #
            else:
                last_state[player_id]  = state_t
                last_action[player_id] = action
                player_id = 3 - player_id    # switch turn

            # ── Training step ─────────────────────────────────────────── #
            if len(replay) >= max(cfg["min_replay"], cfg["batch_size"]):
                beta = beta_schedule(global_step, cfg["per_beta_start"], cfg["per_beta_frames"])
                states, actions, rewards, next_states, dones, idxs, weights = replay.sample(cfg["batch_size"], beta)

                states      = states.to(device)
                actions     = actions.to(device)
                rewards     = rewards.to(device)
                next_states = next_states.to(device)
                dones       = dones.to(device)
                weights     = weights.to(device)

                q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_acts = policy_net(next_states).argmax(1, keepdim=True)
                    target_q  = rewards + (cfg["gamma"] ** cfg["n_steps"]) * \
                                target_net(next_states).gather(1, next_acts).squeeze(1) * (1.0 - dones)

                td_errors = (q_vals - target_q).detach().cpu().numpy()
                replay.update_priorities(idxs, td_errors)

                loss = (weights * F.smooth_l1_loss(q_vals, target_q, reduction="none")).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                if global_step % cfg["target_sync"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            global_step += 1
            obs = obs_next

        env.close()

        # Epsilon decay
        if global_step >= cfg["eps_warmup"]:
            epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])

        # ── Periodic evaluation & checkpointing ───────────────────────── #
        if episode % cfg["eval_every"] == 0:
            recent    = ep_outcomes[-cfg["eval_every"]:]
            wins_tr   = recent.count(1)
            losses_tr = recent.count(0)
            denom     = wins_tr + losses_tr
            train_wr  = wins_tr / denom if denom else 0.0

            policy_net.eval()
            stats = evaluate(policy_net, snapshot_net, cfg["eval_games"], device)
            policy_net.train()

            marker = ""
            if stats["win_rate"] > best_wr:
                best_wr = stats["win_rate"]
                torch.save(policy_net.state_dict(), save_dir / "best_model_switching_gym.pt")
                snapshot_net.load_state_dict(policy_net.state_dict())
                marker = "  ⭐ NEW BEST"

            print(
                f"[ep {episode:>7,}]  "
                f"train_wr={train_wr:.3f}  "
                f"eval_wr={stats['win_rate']:.3f}  "
                f"eval_dr={stats['draw_rate']:.3f}  "
                f"eps={epsilon:.4f}  "
                f"replay={len(replay):,}{marker}"
            )

        if episode % 1_000 == 0:
            torch.save(policy_net.state_dict(), save_dir / "last_model_switching_gym.pt")

    print(f"\nDone. Best eval win-rate vs snapshot: {best_wr:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Connect4 DQN Switching Self-Play (Gymnasium)")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action=argparse.BooleanOptionalAction)
        else:
            p.add_argument(f"--{k}", default=v, type=str if v is None else type(v))
    return vars(p.parse_args())


if __name__ == "__main__":
    train(parse_args())
