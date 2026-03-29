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
#  MinMax Agent                                                        #
# ================================================================== #

class MinMaxAgent(AgentInterface):
    def __init__(self, player_id: int, max_depth: int = 5):
        super().__init__(player_id)
        self.max_depth   = max_depth
        self.opponent_id = 2 if player_id == 1 else 1

        # Evaluation lib — owns all scoring / terminal logic
        self._eval = Evaluation(player_id)

        # Precomputed lazily on first move (board size not known at init)
        self._windows: list | None = None

        # Transposition table: {(board_bytes, current_player): (score, depth)}
        self._tt: dict = {}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_windows(self, rows: int, cols: int) -> None:
        """Build window index list once; reuse for the rest of the game."""
        if self._windows is None:
            self._windows = _build_windows(rows, cols)

    def _order_moves(self, valid: list[int], cols: int) -> list[int]:
        """Sort columns centre-first for better move ordering."""
        center = cols / 2
        return sorted(valid, key=lambda c: abs(center - c))

    def _board_key(self, state) -> tuple:
        """Hashable board representation for the transposition table."""
        return (state.board.tobytes(), state.current_player)

    def _terminal_score(self, game) -> int:
        """
        Return the terminal score once is_terminal_node is True.
        Distinguishes win / loss / draw using game.winning_move.
        """
        if game.winning_move(self.player_id):
            return 100000000000000
        if game.winning_move(self.opponent_id):
            return -100000000000000
        return 0    # Draw — board full, no winner

    def _score_board_fast(self, state) -> int:
        board_td = np.flipud(state.board)   # top-down orientation
        cols     = state.column_count
        center   = cols // 2
        score    = 0

        # ── Centre-column bonus (vectorized) ─────────────────────── #
        player_mask = (board_td == self.player_id)
        score += 3 * int(np.sum(player_mask[:, center]))

        # ── Window scoring via Evaluation.evaluate_window ─────────── #
        for idx in self._windows:
            window = [int(board_td[r][c]) for r, c in idx]
            score += self._eval.evaluate_window(state, window)

        return score

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def select_move(self, game) -> int:
        self._tt.clear()  # Fresh table each root call (avoids stale entries)
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

        # ── Pass 2: forced block (opponent wins next turn) ────────── #
        for col in valid:
            child = game.clone()
            child.current_player = self.opponent_id
            success, row = child.make_move(col)
            if success and child.check_winner(row, col, self.opponent_id):
                return col

        # ── Pass 3: full minimax search ───────────────────────────── #
        best_col   = valid[0]
        best_score = -math.inf

        for col in valid:
            child = game.clone()
            success, row = child.make_move(col)
            if not success:
                continue

            child.switch_player()
            score = self._minimax(child, self.max_depth - 1)

            if score > best_score:
                best_score = score
                best_col   = col

        return best_col

    # ------------------------------------------------------------------ #
    #  Recursive search                                                    #
    # ------------------------------------------------------------------ #

    def _minimax(self, state, depth: int) -> int:
        is_maximizing = (state.current_player == self.player_id)

        # ── Transposition table lookup ────────────────────────────── #
        key = self._board_key(state)
        if key in self._tt:
            cached_score, cached_depth = self._tt[key]
            if cached_depth >= depth:
                return cached_score

        # ── Terminal check via Evaluation ─────────────────────────── #
        if self._eval.is_terminal_node(state):
            result = self._terminal_score(state)
            self._tt[key] = (result, depth)
            return result

        # ── Leaf node: heuristic via optimized scorer ─────────────── #
        if depth == 0:
            result = self._score_board_fast(state)
            self._tt[key] = (result, depth)
            return result

        # Centre-first move ordering at every level
        valid = self._order_moves(self.get_valid_moves(state), state.column_count)

        if is_maximizing:
            best = -math.inf
            for col in valid:
                child = state.clone()
                success, row = child.make_move(col)
                if not success:
                    continue
                if child.check_winner(row, col, self.player_id):
                    result = 100000000000000 + depth    # Win — prefer faster
                    self._tt[key] = (result, depth)
                    return result
                child.switch_player()
                best = max(best, self._minimax(child, depth - 1))
            self._tt[key] = (best, depth)
            return best
        else:
            best = math.inf
            for col in valid:
                child = state.clone()
                success, row = child.make_move(col)
                if not success:
                    continue
                if child.check_winner(row, col, self.opponent_id):
                    result = -100000000000000 - depth   # Loss — prefer slower
                    self._tt[key] = (result, depth)
                    return result
                child.switch_player()
                best = min(best, self._minimax(child, depth - 1))
            self._tt[key] = (best, depth)
            return best