import argparse
import collections
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.connect4_env import Connect4Env            # ← Gymnasium wrapper (unified)
from agents.reinforcement_agent import Connect4Net
from agents.rule_based_agent import RuleBasedAgent

DEFAULTS = dict(
    episodes=60_000,
    batch_size=256,
    replay_capacity=120_000,
    lr=1e-4,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.999_95,
    target_sync=500,
    eval_every=1_000,
    eval_games=200,
    min_replay=5_000,
    save_dir="checkpoints",
    resume=None,
)

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, n):
        s, a, r, ns, d = zip(*random.sample(self.buf, n))
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(ns),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def obs_to_tensor(obs: np.ndarray, player_id: int, valid_moves: list, device) -> torch.Tensor:
    """Convert Gymnasium observation (6×7) to 3-channel tensor [3, 6, 7]."""
    board_f     = np.flipud(obs.astype(np.float32))
    opp_id      = 2 if player_id == 1 else 1
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
    q_np   = q_values.cpu().numpy()
    for col in valid_moves:
        masked[col] = q_np[col]
    return int(np.argmax(masked))


def get_valid_moves(env: Connect4Env) -> list[int]:
    game = env.game
    return [c for c in range(7) if game.is_valid_location(game.board, c)]


def evaluate(policy_net, n_games: int, device) -> float:
    """Win-rate of policy_net (as P1) vs a fresh RuleBasedAgent (P2)."""
    wins = draws = losses = 0
    for _ in range(n_games):
        opponent = RuleBasedAgent(player_id=2)
        env = Connect4Env(opponent_agent=opponent, render_mode=None)
        obs, info = env.reset()
        done = False
        while not done:
            valid = get_valid_moves(env)
            state_t = obs_to_tensor(obs, 1, valid, device)
            with torch.no_grad():
                q_vals = policy_net(state_t.unsqueeze(0)).squeeze(0)
            action = pick_action(q_vals, valid)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        env.close()
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    total = wins + draws + losses
    return wins / total if total else 0.0


def train(cfg):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    policy_net = Connect4Net().to(device)
    target_net = copy.deepcopy(policy_net).to(device)
    target_net.eval()

    if cfg["resume"]:
        policy_net.load_state_dict(torch.load(cfg["resume"], map_location=device))
        print(f"[resume] Loaded weights from {cfg['resume']}")

    optimizer   = optim.Adam(policy_net.parameters(), lr=cfg["lr"])
    replay      = ReplayBuffer(cfg["replay_capacity"])
    epsilon     = cfg["eps_start"]
    best_wr     = -1.0
    global_step = 0
    ep_rewards: list[float] = []
    ep_outcomes: list[int]  = []

    for episode in range(1, cfg["episodes"] + 1):
        # Randomly swap sides each episode so the agent learns both perspectives
        dqn_pid = random.choice([1, 2])
        opp_pid = 3 - dqn_pid

        opponent = RuleBasedAgent(player_id=opp_pid)

        # Connect4Env with opponent built in:
        #   - DQN plays as dqn_pid (always acts via env.step)
        #   - env handles opponent's response automatically after each step
        #   - reward is always from the ACTING player's point of view
        env = Connect4Env(opponent_agent=opponent if dqn_pid == 1 else None, render_mode=None)
        obs, info = env.reset()

        # When dqn_pid == 2, opponent must move first before we get control.
        # We handle this by not using the built-in opponent for dqn_pid==2 and
        # instead manually stepping for the opponent before passing to DQN.
        # Simpler: always play DQN as P1 but flip the board view.
        # Actually the cleanest approach: use Connect4Env without built-in opponent
        # and manually handle both sides, using the env only for board state + win check.
        env.close()

        # Use env in manual mode (no built-in opponent) to support random side assignment
        env = Connect4Env(opponent_agent=None, render_mode=None)
        obs, info = env.reset()

        ep_reward = 0.0
        done      = False
        last_state:  torch.Tensor | None = None
        last_action: int | None          = None

        while not done:
            pid   = info["current_player"]
            valid = get_valid_moves(env)

            if not valid:
                if last_state is not None:
                    replay.push(last_state, last_action, 0.2,
                                obs_to_tensor(obs, dqn_pid, [], device), 1.0)
                    ep_reward += 0.2
                break

            if pid == dqn_pid:
                # DQN's turn
                state_t = obs_to_tensor(obs, dqn_pid, valid, device)
                if random.random() < epsilon:
                    action = random.choice(valid)
                else:
                    with torch.no_grad():
                        action = pick_action(policy_net(state_t.unsqueeze(0)).squeeze(0), valid)

                obs_next, reward_raw, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    # reward_raw is from the env's perspective (acting player = dqn_pid)
                    r = reward_raw if reward_raw != -10 else -1.0  # treat illegal as loss
                    replay.push(state_t, action, r,
                                obs_to_tensor(obs_next, dqn_pid, [], device), 1.0)
                    ep_reward += r
                else:
                    last_state  = state_t
                    last_action = action

                obs = obs_next

            else:
                # Opponent's turn (RuleBasedAgent)
                opp_action = opponent.select_move(env.game)
                obs_next, reward_raw, terminated, truncated, info = env.step(opp_action)
                done = terminated or truncated

                if done:
                    # Opponent won or draw — credit DQN's last action
                    r = -reward_raw if reward_raw > 0 else 0.2   # opp win → DQN loses
                    if last_state is not None:
                        replay.push(last_state, last_action, r,
                                    obs_to_tensor(obs_next, dqn_pid, [], device), 1.0)
                        ep_reward += r
                else:
                    # Intermediate step: push non-terminal transition for DQN's last move
                    if last_state is not None:
                        next_valid = get_valid_moves(env)
                        replay.push(last_state, last_action, 0.0,
                                    obs_to_tensor(obs_next, dqn_pid, next_valid, device), 0.0)
                        last_state  = None
                        last_action = None

                obs = obs_next

            # Training step
            if len(replay) >= max(cfg["min_replay"], cfg["batch_size"]):
                states, actions, rewards, next_states, dones = replay.sample(cfg["batch_size"])
                states, actions = states.to(device), actions.to(device)
                rewards, next_states, dones = rewards.to(device), next_states.to(device), dones.to(device)

                q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_acts = policy_net(next_states).argmax(1, keepdim=True)
                    target_q  = rewards + cfg["gamma"] * target_net(next_states).gather(1, next_acts).squeeze(1) * (1.0 - dones)

                loss = nn.SmoothL1Loss()(q_vals, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                if global_step % cfg["target_sync"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])
            global_step += 1

        env.close()

        ep_rewards.append(ep_reward)
        ep_outcomes.append(1 if ep_reward > 0.5 else 0 if ep_reward < -0.5 else -1)

        if episode % cfg["eval_every"] == 0:
            recent    = ep_outcomes[-cfg["eval_every"]:]
            wins_tr   = recent.count(1)
            losses_tr = recent.count(0)
            train_wr  = wins_tr / (wins_tr + losses_tr) if (wins_tr + losses_tr) > 0 else 0.0

            policy_net.eval()
            eval_wr = evaluate(policy_net, cfg["eval_games"], device)
            policy_net.train()

            marker = ""
            if train_wr > best_wr:
                best_wr = train_wr
                torch.save(policy_net.state_dict(), save_dir / "best_model_rule_based_gym.pt")
                marker = "  ⭐ NEW BEST"

            print(
                f"[ep {episode:>7,}]  "
                f"avg_r={np.mean(ep_rewards[-cfg['eval_every']:]):+.3f}  "
                f"eps={epsilon:.4f}  "
                f"replay={len(replay):,}  "
                f"train_wr={train_wr:.3f}  "
                f"eval_wr={eval_wr:.3f}{marker}"
            )

        torch.save(policy_net.state_dict(), save_dir / "last_model_rule_based_gym.pt")

    print(f"\nDone. Best win-rate: {best_wr:.3f}")


def parse_args():
    p = argparse.ArgumentParser(description="Connect4 DQN vs RuleBasedAgent (Gymnasium)")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", default=v, type=str if v is None else type(v))
    return vars(p.parse_args())


if __name__ == "__main__":
    train(parse_args())
