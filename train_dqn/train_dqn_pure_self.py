"""
DQN Training Script for Connect4 — Pure Self-Play
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

DEFAULTS = dict(
    episodes=30_000, 
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
        game = Connect4Game()
        done = False
        
        # Track previous states and actions for BOTH players during self-play
        # pid 1 -> Player 1, pid 2 -> Player 2
        last_state = {1: None, 2: None}
        last_action = {1: None, 2: None}
        
        # Tracking reward from Player 1's perspective for logging purposes
        ep_reward = 0.0 

        while not done:
            pid = game.current_player
            opp_pid = 3 - pid 
            valid = game.get_valid_locations()

            if not valid:
                break

            # Get state from the perspective of the CURRENT player
            state_t = board_to_tensor(game.board, pid, valid, device)

            # Both players use the exact same policy_net
            if random.random() < epsilon:
                col = random.choice(valid)
            else:
                with torch.no_grad():
                    col = pick_action(policy_net(state_t.unsqueeze(0)).squeeze(0), valid)

            # Before making a move, the PREVIOUS player's turn is complete 
            # because we can now see the state they left us in.
            if last_state[opp_pid] is not None:
                next_state_opp = board_to_tensor(game.board, opp_pid, valid, device)
                replay.push(last_state[opp_pid], last_action[opp_pid], 0.0, next_state_opp, 0.0)

            _, row = game.make_move(col)

            # Terminal Conditions
            if row is not None and game.check_winner(row, col, pid):
                # Current player WON
                replay.push(state_t, col, 1.0, board_to_tensor(game.board, pid, [], device), 1.0)
                
                # Opponent LOST
                if last_state[opp_pid] is not None:
                    replay.push(last_state[opp_pid], last_action[opp_pid], -1.0, board_to_tensor(game.board, opp_pid, [], device), 1.0)
                
                ep_reward = 1.0 if pid == 1 else -1.0
                done = True
                
            elif game.check_draw():
                # Both players DRAW
                replay.push(state_t, col, 0.2, board_to_tensor(game.board, pid, [], device), 1.0)
                if last_state[opp_pid] is not None:
                    replay.push(last_state[opp_pid], last_action[opp_pid], 0.2, board_to_tensor(game.board, opp_pid, [], device), 1.0)
                
                ep_reward = 0.2
                done = True
                
            else:
                # Update trackers and switch turn
                last_state[pid] = state_t
                last_action[pid] = col
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

            global_step += 1

        epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])

        # Logging tracking
        ep_rewards.append(ep_reward)
        ep_outcomes.append(1 if ep_reward > 0.5 else 0 if ep_reward < -0.5 else -1)

        if episode % 1000 == 0:
            recent = ep_outcomes[-1000:]
            wins = recent.count(1)   # Player 1 Wins
            losses = recent.count(0) # Player 2 Wins
            
            # Since P1 goes first, an equal win rate is ~55% for P1 due to Connect 4 mechanics.
            train_wr = wins / (wins + losses) if (wins + losses) > 0 else 0.0
            
            marker = ""
            if train_wr > best_wr:
                best_wr = train_wr
                torch.save(policy_net.state_dict(), save_dir / "best_model_pure_self.pt")
                marker = "  ⭐ NEW BEST (First-Mover WR)"
                
            print(f"[ep {episode:>7,}]  P1_wr={train_wr:.3f}  "
                  f"eps={epsilon:.4f}  replay={len(replay):,}{marker}")

        torch.save(policy_net.state_dict(), save_dir / "last_model_pure_self.pt")

    print(f"\nDone. Best P1 win-rate: {best_wr:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", default=v, type=str if v is None else type(v))
    return vars(p.parse_args())


if __name__ == "__main__":
    train(parse_args())