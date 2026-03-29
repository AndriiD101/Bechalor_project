from agents.agents_interface import AgentInterface
from game.connect4 import Connect4Game
import numpy as np


# ------------------------------------------------------------------ #
#  Module-level window cache                                           #
# ------------------------------------------------------------------ #

# Window coordinates are the same for every Connect-4 game (6×7 board).
# Pre-computing them once at import time means _all_windows() is never
# called inside a hot path again.

def _build_window_cache(rows: int = 6, cols: int = 7) -> tuple:
    """
    Returns four numpy arrays of shape (N, 4, 2):
        horiz_wins, vert_wins, diag_wins, anti_wins
    Each row is one window; each entry is a (row, col) pair.

    Also returns a flat list of all windows combined for callers that
    do not need to distinguish direction.
    """
    horiz, vert, diag, anti = [], [], [], []

    for r in range(rows):
        for c in range(cols - 3):
            horiz.append([(r, c + i) for i in range(4)])

    for r in range(rows - 3):
        for c in range(cols):
            vert.append([(r + i, c) for i in range(4)])

    for r in range(rows - 3):
        for c in range(cols - 3):
            diag.append([(r + i, c + i) for i in range(4)])

    for r in range(3, rows):
        for c in range(cols - 3):
            anti.append([(r - i, c + i) for i in range(4)])

    to_np = lambda lst: np.array(lst, dtype=np.int8)   # shape (N,4,2)
    all_wins = horiz + vert + diag + anti
    return to_np(horiz), to_np(vert), to_np(diag), to_np(anti), to_np(all_wins)


_HORIZ_W, _VERT_W, _DIAG_W, _ANTI_W, _ALL_W = _build_window_cache()

# Pre-extract row/col index arrays for vectorised board look-ups.
# Shape: (N_windows, 4)  — used as board[_W_ROWS, _W_COLS]
_W_ROWS = _ALL_W[:, :, 0].astype(np.intp)   # (N, 4)
_W_COLS = _ALL_W[:, :, 1].astype(np.intp)   # (N, 4)


def _extract_windows(board: np.ndarray) -> np.ndarray:
    """
    Return all 69 windows of size 4 as a (69, 4) int8 array in one
    vectorised gather — no Python loops at all.
    """
    return board[_W_ROWS, _W_COLS]          # shape (69, 4)


# ------------------------------------------------------------------ #
#  Agent                                                               #
# ------------------------------------------------------------------ #

class RuleBasedAgent(AgentInterface):
    """
    Rule-based Connect-4 agent.

    Priority order (unchanged from original):
      1.  Win immediately
      2.  Block opponent's immediate win
      3.  Never play below a game-ending space
      4.  Build a seven trap (double threat)
      5.  Block opponent's seven trap
      6.  Odd-even parity strategy
      7.  Build open-ended three-in-a-row
      8.  Block opponent's open-ended three-in-a-row
      9.  Block opponent's two-in-a-row (early threat disruption)
      10. Build our own two-in-a-row
      11. Center column dominance
      12. Prefer columns nearest center

    Speed improvements over the original:
      FIX 1 — Window coordinates are pre-computed once at module import
               time and stored as numpy arrays; _all_windows() is never
               called inside any hot path.
      FIX 2 — All window scoring (threats, two-threats, parity) is done
               with vectorised numpy operations instead of Python loops
               over individual cells.
      FIX 3 — Simulated moves use a direct board copy + incremental
               drop/undo instead of a full Connect4Game.clone() for every
               candidate column in every strategy.
      FIX 4 — A single _analyse_board() pass computes every count needed
               by strategies 4-10 in one sweep, so the window array is
               read only once per candidate move instead of once per
               strategy per candidate move.
      FIX 5 — _immediate_win and _filter_game_ending_moves use the fast
               incremental win checker so we never scan the whole board.
      FIX 6 — Valid-move detection reads the top cell of each column
               directly instead of iterating through rows.
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def select_move(self, game: Connect4Game) -> int:
        board = game.board
        rows, cols = game.row_count, game.column_count
        center = cols // 2

        # FIX 6: Fast valid-move detection via top-row check
        valid_moves = self._fast_valid_moves(board, rows, cols)
        if not valid_moves:
            return -1

        # 1. Win immediately
        win = self._immediate_win(board, rows, cols, valid_moves, self.player_id)
        if win is not None:
            return win

        # 2. Block opponent's immediate win
        block = self._immediate_win(board, rows, cols, valid_moves, self.opponent_id)
        if block is not None:
            return block

        # 3. Remove moves that set up the opponent above us
        safe_moves = self._filter_game_ending_moves(board, rows, cols, valid_moves)
        if not safe_moves:
            safe_moves = valid_moves

        # FIX 4: Pre-compute all per-column analysis in a single pass
        # col_stats[col] = dict with all counts needed for rules 4-10
        col_stats = self._analyse_candidates(board, rows, cols, safe_moves)

        # 4. Build a seven trap for ourselves
        for col in safe_moves:
            if col_stats[col]['my_threats'] >= 2:
                return col

        # 5. Block opponent's seven trap
        for col in safe_moves:
            if col_stats[col]['op_threats'] >= 2:
                return col

        # 6. Parity (odd-even) strategy
        target_parity = 0 if self.player_id == 1 else 1
        best_parity_col = max(
            safe_moves,
            key=lambda c: col_stats[c]['my_parity_threats'],
            default=None
        )
        if best_parity_col is not None and col_stats[best_parity_col]['my_parity_threats'] > 0:
            return best_parity_col

        # 7. Build open-ended three-in-a-row
        for col in safe_moves:
            if col_stats[col]['my_open_three']:
                return col

        # 8. Block opponent's open-ended three-in-a-row
        for col in safe_moves:
            if col_stats[col]['op_open_three']:
                return col

        # 9. Block opponent's two-in-a-row (pick the move that removes the most)
        original_op_twos = self._count_two_threats_vec(board, self.opponent_id, self.player_id)
        best_block_col = None
        best_remaining = original_op_twos
        for col in safe_moves:
            remaining = col_stats[col]['op_twos']
            if remaining < best_remaining:
                best_remaining = remaining
                best_block_col = col
        if best_block_col is not None:
            return best_block_col

        # 10. Build our own two-in-a-row
        best_build_col = max(
            safe_moves,
            key=lambda c: col_stats[c]['my_twos'],
            default=None
        )
        if best_build_col is not None and col_stats[best_build_col]['my_twos'] > 0:
            return best_build_col

        # 11. Center column
        if center in safe_moves:
            return center

        # 12. Nearest to center
        return min(safe_moves, key=lambda c: abs(c - center))

    # ------------------------------------------------------------------ #
    #  FIX 4: Single-pass candidate analysis                              #
    # ------------------------------------------------------------------ #

    def _analyse_candidates(self, board: np.ndarray, rows: int, cols: int,
                             candidates: list[int]) -> dict:
        """
        For each candidate column, drop a piece, compute every metric
        needed by rules 4-10 in one vectorised pass, then undo the drop.

        Returns dict: col -> {my_threats, op_threats, my_parity_threats,
                               my_open_three, op_open_three,
                               my_twos, op_twos}
        """
        target_parity = 0 if self.player_id == 1 else 1
        stats = {}

        for col in candidates:
            row = self._drop_row(board, rows, col)
            if row == -1:
                continue

            # Simulate: place our piece
            board[row, col] = self.player_id
            windows = _extract_windows(board)   # (69, 4) — FIX 1+2

            my  = (windows == self.player_id).sum(axis=1)
            op  = (windows == self.opponent_id).sum(axis=1)
            em  = (windows == 0).sum(axis=1)

            # --- Threats (3 mine + 1 empty, empty cell is playable) ---
            my_threat_mask = (my == 3) & (op == 0) & (em == 1)
            op_threat_mask = (my == 0) & (op == 3) & (em == 1)

            my_threats = self._count_playable_threats(board, windows, my_threat_mask, self.player_id)
            op_threats = self._count_playable_threats(board, windows, op_threat_mask, self.opponent_id)

            # --- Parity threats ---
            my_parity = self._count_parity_threats_vec(board, windows, my_threat_mask, target_parity)

            # --- Open-ended three (horizontal only, 5-wide window) ---
            my_open_three  = self._has_open_three(board, rows, cols, self.player_id)
            op_open_three  = self._has_open_three(board, rows, cols, self.opponent_id)

            # --- Two-in-a-row counts ---
            my_twos = int(np.sum((my == 2) & (op == 0) & (em == 2)))
            op_twos = self._count_two_threats_vec(board, self.opponent_id, self.player_id)

            board[row, col] = 0   # undo

            stats[col] = {
                'my_threats':        my_threats,
                'op_threats':        op_threats,
                'my_parity_threats': my_parity,
                'my_open_three':     my_open_three,
                'op_open_three':     op_open_three,
                'my_twos':           my_twos,
                'op_twos':           op_twos,
            }

        return stats

    # ------------------------------------------------------------------ #
    #  FIX 5: Immediate-win check (incremental, no full-board scan)       #
    # ------------------------------------------------------------------ #

    def _immediate_win(self, board: np.ndarray, rows: int, cols: int,
                       valid_moves: list[int], player: int) -> int | None:
        for col in valid_moves:
            row = self._drop_row(board, rows, col)
            if row == -1:
                continue
            board[row, col] = player
            won = self._check_winner_fast(board, rows, cols, row, col, player)
            board[row, col] = 0
            if won:
                return col
        return None

    # ------------------------------------------------------------------ #
    #  FIX 3+5: Filter game-ending moves (no Clone, fast win check)      #
    # ------------------------------------------------------------------ #

    def _filter_game_ending_moves(self, board: np.ndarray, rows: int, cols: int,
                                   moves: list[int]) -> list[int]:
        """
        Drop any column where the cell directly above our landing position
        could immediately be won by either player.
        """
        safe = []
        for col in moves:
            row = self._drop_row(board, rows, col)
            if row == -1:
                continue
            above = row + 1
            if above >= rows:
                safe.append(col)
                continue

            dangerous = False
            for player in (self.player_id, self.opponent_id):
                board[above, col] = player
                if self._check_winner_fast(board, rows, cols, above, col, player):
                    dangerous = True
                board[above, col] = 0
                if dangerous:
                    break

            if not dangerous:
                safe.append(col)
        return safe

    # ------------------------------------------------------------------ #
    #  FIX 2: Vectorised threat / window counters                         #
    # ------------------------------------------------------------------ #

    def _count_playable_threats(self, board: np.ndarray,
                                 windows: np.ndarray,
                                 mask: np.ndarray,
                                 player: int) -> int:
        """
        Among windows selected by `mask`, count those whose single empty
        cell sits on a surface that can actually be played (i.e. the cell
        below it is occupied or it is on the bottom row).  FIX 1+2.
        """
        if not mask.any():
            return 0
        count = 0
        for idx in np.where(mask)[0]:
            window_cells = windows[idx]           # length-4 array of piece values
            empty_pos = int(np.argmax(window_cells == 0))
            er = int(_W_ROWS[idx, empty_pos])
            ec = int(_W_COLS[idx, empty_pos])
            if er == 0 or board[er - 1, ec] != 0:
                count += 1
        return count

    def _count_parity_threats_vec(self, board: np.ndarray,
                                   windows: np.ndarray,
                                   threat_mask: np.ndarray,
                                   target_parity: int) -> int:
        """Count my playable threats whose empty cell row matches target_parity."""
        if not threat_mask.any():
            return 0
        count = 0
        for idx in np.where(threat_mask)[0]:
            window_cells = windows[idx]
            empty_pos = int(np.argmax(window_cells == 0))
            er = int(_W_ROWS[idx, empty_pos])
            ec = int(_W_COLS[idx, empty_pos])
            if (er == 0 or board[er - 1, ec] != 0) and er % 2 == target_parity:
                count += 1
        return count

    def _count_two_threats_vec(self, board: np.ndarray,
                                player: int, opponent: int) -> int:
        """
        Count windows where `player` has exactly 2 pieces, no opponent
        pieces, and 2 empty cells — fully vectorised.  FIX 1+2.
        """
        windows = _extract_windows(board)
        my = (windows == player).sum(axis=1)
        op = (windows == opponent).sum(axis=1)
        em = (windows == 0).sum(axis=1)
        return int(np.sum((my == 2) & (op == 0) & (em == 2)))

    def _has_open_three(self, board: np.ndarray,
                         rows: int, cols: int, player: int) -> bool:
        """
        Detect a horizontal [empty, p, p, p, empty] pattern where both
        flanking empty cells are immediately playable.  FIX 1 (no clone).
        """
        for r in range(rows):
            for c in range(cols - 4):
                w = board[r, c:c + 5]
                if (int((w == player).sum()) == 3
                        and int(w[0]) == 0 and int(w[4]) == 0):
                    left_ok  = r == 0 or board[r - 1, c]     != 0
                    right_ok = r == 0 or board[r - 1, c + 4] != 0
                    if left_ok and right_ok:
                        return True
        return False

    # ------------------------------------------------------------------ #
    #  FIX 1: Incremental win checker (O(1) vs O(rows*cols))             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_winner_fast(board: np.ndarray, rows: int, cols: int,
                            row: int, col: int, player: int) -> bool:
        """Check only the four directions through the last-placed piece."""
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  FIX 3+6: Lightweight board helpers (no game clones)               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fast_valid_moves(board: np.ndarray, rows: int, cols: int) -> list[int]:
        """A column is valid iff its top cell is empty — O(cols)."""
        return [c for c in range(cols) if board[rows - 1, c] == 0]

    @staticmethod
    def _drop_row(board: np.ndarray, rows: int, col: int) -> int:
        """Return the row a piece would land in, or -1 if column is full."""
        for row in range(rows):
            if board[row, col] == 0:
                return row
        return -1