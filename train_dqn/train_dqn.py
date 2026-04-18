"""
DQN Training Script for Connect4 — vs RuleBasedAgent
"""

import argparse
import collections
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.connect4 import Connect4Game
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


def board_to_tensor(board, player_id, valid_moves, device):
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


def pick_action(q_values, valid_moves):
    masked = np.full(7, -1e9, dtype=np.float32)
    q_np = q_values.cpu().numpy()
    for col in valid_moves:
        masked[col] = q_np[col]
    return int(np.argmax(masked))


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    policy_net = Connect4Net().to(device)
    target_net = copy.deepcopy(policy_net).to(device)
    target_net.eval()

    if cfg["resume"]:
        policy_net.load_state_dict(torch.load(cfg["resume"], map_location=device))
        print(f"[resume] Loaded weights from {cfg['resume']}")

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg["lr"])
    replay = ReplayBuffer(cfg["replay_capacity"])
    epsilon = cfg["eps_start"]
    best_wr, global_step = -1.0, 0
    ep_rewards, ep_outcomes = [], []

    for episode in range(1, cfg["episodes"] + 1):
        dqn_pid = random.choice([1, 2])
        opp_pid = 3 - dqn_pid
        opponent = RuleBasedAgent(player_id=opp_pid)
        game = Connect4Game()
        ep_reward, done = 0.0, False
        last_state, last_action = None, None

        while not done:
            pid = game.current_player
            valid = game.get_valid_locations()

            if not valid:
                if last_state is not None:
                    replay.push(last_state, last_action, 0.2,
                                board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                    ep_reward += 0.2
                break

            if pid == dqn_pid:
                state_t = board_to_tensor(game.board, dqn_pid, valid, device)
                if random.random() < epsilon:
                    col = random.choice(valid)
                else:
                    with torch.no_grad():
                        col = pick_action(policy_net(state_t.unsqueeze(0)).squeeze(0), valid)

                _, row = game.make_move(col)

                if row is not None and game.check_winner(row, col, dqn_pid):
                    replay.push(state_t, col, 1.0,
                                board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                    ep_reward += 1.0
                    done = True
                elif game.check_draw():
                    replay.push(state_t, col, 0.2,
                                board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                    ep_reward += 0.2
                    done = True
                else:
                    last_state, last_action = state_t, col
                    game.switch_player()

            else:
                col = opponent.select_move(game)
                _, row = game.make_move(col)

                if row is not None and game.check_winner(row, col, opp_pid):
                    if last_state is not None:
                        replay.push(last_state, last_action, -1.0,
                                    board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                        ep_reward -= 1.0
                    done = True
                elif game.check_draw():
                    if last_state is not None:
                        replay.push(last_state, last_action, 0.2,
                                    board_to_tensor(game.board, dqn_pid, [], device), 1.0)
                        ep_reward += 0.2
                    done = True
                else:
                    if last_state is not None:
                        next_valid = game.get_valid_locations()
                        replay.push(last_state, last_action, 0.0,
                                    board_to_tensor(game.board, dqn_pid, next_valid, device), 0.0)
                        last_state = last_action = None
                    game.switch_player()

            # Training step
            if len(replay) >= max(cfg["min_replay"], cfg["batch_size"]):
                states, actions, rewards, next_states, dones = replay.sample(cfg["batch_size"])
                states, actions, rewards = states.to(device), actions.to(device), rewards.to(device)
                next_states, dones = next_states.to(device), dones.to(device)

                q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_acts = policy_net(next_states).argmax(1, keepdim=True)
                    target_q = rewards + cfg["gamma"] * target_net(next_states).gather(1, next_acts).squeeze(1) * (1.0 - dones)

                loss = nn.SmoothL1Loss()(q_vals, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                if global_step % cfg["target_sync"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])
            global_step += 1

        ep_rewards.append(ep_reward)
        ep_outcomes.append(1 if ep_reward > 0.5 else 0 if ep_reward < -0.5 else -1)

        if episode % 500 == 0:
            recent = ep_outcomes[-500:]
            wins, losses = recent.count(1), recent.count(0)
            train_wr = wins / (wins + losses) if (wins + losses) > 0 else 0.0
            marker = ""
            if train_wr > best_wr:
                best_wr = train_wr
                torch.save(policy_net.state_dict(), save_dir / "best_model.pt")
                marker = "  ⭐ NEW BEST"
            print(f"[ep {episode:>7,}]  avg_r={np.mean(ep_rewards[-500:]):+.3f}  "
                  f"eps={epsilon:.4f}  replay={len(replay):,}  train_wr={train_wr:.3f}{marker}")

        torch.save(policy_net.state_dict(), save_dir / "last_model.pt")

    print(f"\nDone. Best win-rate: {best_wr:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", default=v, type=str if v is None else type(v))
    return vars(p.parse_args())


if __name__ == "__main__":
    train(parse_args())