import math
import numpy as np

from agents.agents_interface import AgentInterface
from agents.evaluation import Evaluation


# ================================================================== #
#  Window index pre-computation                                        #
# ================================================================== #

def _build_windows(rows: int, cols: int) -> list[list[tuple[int, int]]]:
    windows = []

    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            windows.append([(r, c + i) for i in range(4)])

    # Vertical
    for c in range(cols):
        for r in range(rows - 3):
            windows.append([(r + i, c) for i in range(4)])

    # Diagonal /
    for r in range(rows - 3):
        for c in range(cols - 3):
            windows.append([(r + i, c + i) for i in range(4)])

    # Diagonal \
    for r in range(3, rows):
        for c in range(cols - 3):
            windows.append([(r - i, c + i) for i in range(4)])

    return windows


# ================================================================== #
#  Alpha-Beta Agent                                                    #
# ================================================================== #

class AlphaBetaAgent(AgentInterface):
    def __init__(self, player_id: int, max_depth: int = 5):
        super().__init__(player_id)
        self.max_depth   = max_depth
        self.opponent_id = 2 if player_id == 1 else 1

        self._eval    = Evaluation(player_id)
        self._windows: list | None = None
        self._tt: dict = {}

    # ------------------------------------------------------------------ #
    #  Internal helpers  (identical to MinMaxAgent)                        #
    # ------------------------------------------------------------------ #

    def _ensure_windows(self, rows: int, cols: int) -> None:
        if self._windows is None:
            self._windows = _build_windows(rows, cols)

    def _order_moves(self, valid: list[int], cols: int) -> list[int]:
        """Sort columns centre-first — critical for pruning efficiency."""
        center = cols / 2
        return sorted(valid, key=lambda c: abs(center - c))

    def _board_key(self, state) -> tuple:
        return (state.board.tobytes(), state.current_player)

    def _terminal_score(self, game) -> int:
        if game.winning_move(self.player_id):
            return 100000000000000
        if game.winning_move(self.opponent_id):
            return -100000000000000
        return 0

    def _score_board_fast(self, state) -> int:
        board_td = np.flipud(state.board)
        cols     = state.column_count
        center   = cols // 2
        score    = 0

        player_mask = (board_td == self.player_id)
        score += 3 * int(np.sum(player_mask[:, center]))

        for idx in self._windows:
            window = [int(board_td[r][c]) for r, c in idx]
            score += self._eval.evaluate_window(state, window)

        return score

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def select_move(self, game) -> int:
        self._tt.clear()
        self._ensure_windows(game.row_count, game.column_count)

        valid = self.get_valid_moves(game)
        if not valid:
            return -1

        valid = self._order_moves(valid, game.column_count)

        # ── Pass 1: instant win ───────────────────────────────────── #
        for col in valid:
            child = game.clone()
            success, row = child.make_move(col)
            if success and child.check_winner(row, col, self.player_id):
                return col

        # ── Pass 2: forced block ──────────────────────────────────── #
        for col in valid:
            child = game.clone()
            child.current_player = self.opponent_id
            success, row = child.make_move(col)
            if success and child.check_winner(row, col, self.opponent_id):
                return col

        # ── Pass 3: alpha-beta search ─────────────────────────────── #
        best_col   = valid[0]
        best_score = -math.inf
        alpha      = -math.inf
        beta       = math.inf

        for col in valid:
            child = game.clone()
            success, row = child.make_move(col)
            if not success:
                continue

            child.switch_player()
            score = self._alphabeta(child, self.max_depth - 1, alpha, beta)

            if score > best_score:
                best_score = score
                best_col   = col

            # Tighten alpha at the root so siblings are pruned early
            alpha = max(alpha, best_score)

        return best_col

    # ------------------------------------------------------------------ #
    #  Recursive search                                                    #
    # ------------------------------------------------------------------ #

    def _alphabeta(self, state, depth: int, alpha: float, beta: float) -> int:
        is_maximizing = (state.current_player == self.player_id)

        # ── Transposition table lookup ────────────────────────────── #
        key = self._board_key(state)
        if key in self._tt:
            cached_score, cached_depth = self._tt[key]
            if cached_depth >= depth:
                return cached_score

        # ── Terminal check ────────────────────────────────────────── #
        if self._eval.is_terminal_node(state):
            result = self._terminal_score(state)
            self._tt[key] = (result, depth)
            return result

        # ── Leaf node: heuristic score ────────────────────────────── #
        if depth == 0:
            result = self._score_board_fast(state)
            self._tt[key] = (result, depth)
            return result

        valid = self._order_moves(self.get_valid_moves(state), state.column_count)

        if is_maximizing:
            best = -math.inf
            for col in valid:
                child = state.clone()
                success, row = child.make_move(col)
                if not success:
                    continue

                # Immediate win — no need to search deeper
                if child.check_winner(row, col, self.player_id):
                    result = 100000000000000 + depth
                    self._tt[key] = (result, depth)
                    return result

                child.switch_player()
                best = max(best, self._alphabeta(child, depth - 1, alpha, beta))
                alpha = max(alpha, best)

                # β-cutoff: minimiser won't allow this branch
                if alpha >= beta:
                    break

            self._tt[key] = (best, depth)
            return best

        else:
            best = math.inf
            for col in valid:
                child = state.clone()
                success, row = child.make_move(col)
                if not success:
                    continue

                # Immediate loss — no need to search deeper
                if child.check_winner(row, col, self.opponent_id):
                    result = -100000000000000 - depth
                    self._tt[key] = (result, depth)
                    return result

                child.switch_player()
                best = min(best, self._alphabeta(child, depth - 1, alpha, beta))
                beta = min(beta, best)

                # α-cutoff: maximiser won't allow this branch
                if alpha >= beta:
                    break

            self._tt[key] = (best, depth)
            return best