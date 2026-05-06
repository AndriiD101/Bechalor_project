"""
Connect4Env – a Gymnasium-compatible environment that wraps Connect4Game.

Usage (identical to any other Gymnasium environment):

    import gymnasium as gym
    from game.connect4_env import Connect4Env

    env = Connect4Env()                # or Connect4Env(opponent_agent=some_agent)
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()   # replace with your agent
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()

Observation space:
    Box(0, 2, shape=(6, 7), dtype=np.int8)
    0 = empty, 1 = player-1's piece, 2 = player-2's piece

Action space:
    Discrete(7)  – column index (0-6)

Reward:
    +1  if the acting player wins
    -1  if the acting player loses (only when opponent_agent is set)
     0  draw or step without terminal
    -10 illegal move (episode ends)

Info dict keys:
    current_player  – whose turn it is (1 or 2)
    winner          – 0 (no winner yet), 1, or 2
    illegal_move    – True when an invalid column was chosen
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game.connect4 import Connect4Game


class Connect4Env(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, opponent_agent=None, render_mode: str | None = None):
        super().__init__()

        self.opponent_agent = opponent_agent
        self.render_mode = render_mode

        # ── Spaces ──────────────────────────────────────────────────── #
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=(6, 7),
            dtype=np.int8,
        )
        self.action_space = spaces.Discrete(7)

        # ── Internal state (populated by reset) ─────────────────────── #
        self._game: Connect4Game | None = None

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._game = Connect4Game()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode in ("human", "ansi"):
            self._render_text()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._game is not None, "Call reset() before step()."

        acting_player = self._game.current_player

        # ── Illegal move guard ───────────────────────────────────────── #
        if not self._game.is_valid_location(self._game.board, action):
            obs = self._get_obs()
            info = self._get_info()
            info["illegal_move"] = True
            return obs, -10.0, True, False, info

        # ── Apply action ─────────────────────────────────────────────── #
        success, row = self._game.make_move(action)

        if self.render_mode in ("human", "ansi"):
            self._render_text()

        # ── Check terminal after agent's move ────────────────────────── #
        if self._game.check_winner(row, action, acting_player):
            reward = 1.0
            info = self._get_info(winner=acting_player)
            return self._get_obs(), reward, True, False, info

        if self._game.check_draw():
            info = self._get_info(winner=0)
            return self._get_obs(), 0.0, True, False, info

        # ── Switch to next player ─────────────────────────────────────── #
        self._game.switch_player()

        # ── Opponent's turn (optional) ────────────────────────────────── #
        if self.opponent_agent is not None:
            opp_col = self.opponent_agent.select_move(self._game)

            if opp_col == -1 or not self._game.is_valid_location(self._game.board, opp_col):
                # Opponent has no moves or made an illegal move → draw/win
                info = self._get_info(winner=0)
                return self._get_obs(), 0.0, True, False, info

            _, opp_row = self._game.make_move(opp_col)

            if self.render_mode in ("human", "ansi"):
                self._render_text()

            if self._game.check_winner(opp_row, opp_col, self._game.current_player):
                # Opponent wins → acting player loses
                info = self._get_info(winner=self._game.current_player)
                return self._get_obs(), -1.0, True, False, info

            if self._game.check_draw():
                info = self._get_info(winner=0)
                return self._get_obs(), 0.0, True, False, info

            self._game.switch_player()

        return self._get_obs(), 0.0, False, False, self._get_info()

    def render(self) -> None:
        if self._game is None:
            return
        self._render_text()

    def close(self) -> None:
        self._game = None

    @property
    def game(self) -> Connect4Game:
        return self._game

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        return self._game.board.astype(np.int8)

    def _get_info(self, winner: int = 0) -> dict:
        return {
            "current_player": self._game.current_player,
            "winner": winner,
            "illegal_move": False,
        }

    def _render_text(self) -> None:
        print(np.flipud(self._game.board).astype(int))
        print()
