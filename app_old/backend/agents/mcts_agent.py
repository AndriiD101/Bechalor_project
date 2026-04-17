import math
import time
import random
import numpy as np

from agents.agents_interface import AgentInterface
from game.connect4 import Connect4Game


# ================================================================== #
#  Node                                                                #
# ================================================================== #

class MCTSNode:
    __slots__ = ('game_state', 'parent', 'move', 'children',
                 'visits', 'wins', 'available_moves', 'piece')

    def __init__(self, game_state: Connect4Game, piece: int,
                 parent: 'MCTSNode | None' = None, move: int | None = None):
        self.game_state     = game_state.clone()
        self.parent         = parent
        self.move           = move          # Column that led to this node
        self.children:  list['MCTSNode'] = []
        self.visits         = 0
        self.wins           = 0.0
        self.piece          = piece         # Player who just moved into this node
        self.available_moves: list[int] = game_state.get_valid_locations()

    # ── UCT selection (best child by UCB1) ────────────────────────── #
    def selection(self) -> 'MCTSNode':
        """Return the child with the highest UCT value."""
        log_visits = math.log(self.visits)
        return max(
            self.children,
            key=lambda c: (c.wins / c.visits)
                          + math.sqrt(2 * log_visits / c.visits)
        )

    def expand(self, move: int, new_state: Connect4Game) -> 'MCTSNode':
        """Create and register a child node for the given move."""
        child = MCTSNode(
            game_state=new_state,
            piece=new_state.current_player,  # player whose turn it is next
            parent=self,
            move=move
        )
        self.available_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        self.wins   += result
        self.visits += 1

    def best_move_child(self) -> 'MCTSNode':
        """Return the child with the best win ratio (used at root)."""
        return max(self.children, key=lambda c: c.wins / c.visits)


# ================================================================== #
#  Agent                                                               #
# ================================================================== #

class MCTSAgent(AgentInterface):
    def __init__(self, player_id: int,
                 max_iterations: int = 20000,
                 timeout: float = 2.0):
        super().__init__(player_id)
        self.opponent_id    = 2 if player_id == 1 else 1
        self.max_iterations = max_iterations
        # self.timeout        = timeout
        self._current_node: MCTSNode | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def select_move(self, game: Connect4Game) -> int:
        # ── Tree reuse: advance root to the current board state ───── #
        if self._current_node is None:
            self._current_node = MCTSNode(game_state=game, piece=self.player_id)

        # Opponent may have played a move we haven't seen — find or
        # create the matching child node.
        self._current_node = self._get_or_create_node(
            self._current_node, game, game.current_player
        )

        # ── Run MCTS ─────────────────────────────────────────────── #
        root = self._current_node
        # start = time.perf_counter()

        for _ in range(self.max_iterations):
            node  = root
            state = node.game_state.clone()

            # 1. Selection — descend via UCT until expandable/terminal
            while node.available_moves == [] and node.children != []:
                node = node.selection()
                state.make_move(node.move)
                state.switch_player()

            # 2. Expansion — add one new child for an untried move
            if node.available_moves:
                col = random.choice(node.available_moves)
                state.make_move(col)
                state.switch_player()
                node = node.expand(col, state)

            # 3. Rollout — play out to terminal with guided moves
            result = self._rollout(state)

            # 4. Backpropagation
            self._backpropagate(node, result)

            # if time.perf_counter() - start > self.timeout:
            #     break

        # ── Pick best child by win ratio ──────────────────────────── #
        best = root.best_move_child()

        # Advance persistent root to the chosen child
        best.parent      = None
        self._current_node = best
        return best.move

    # ------------------------------------------------------------------ #
    #  Tree reuse helper                                                   #
    # ------------------------------------------------------------------ #

    def _get_or_create_node(self, node: MCTSNode,
                             game: Connect4Game,
                             piece: int) -> MCTSNode:
        """
        Search immediate children for a node whose board matches `game`.
        Falls back to a fresh node if no match is found (e.g. the opponent
        played an unexplored move).
        """
        for child in node.children:
            if self._boards_equal(child.game_state.board, game.board):
                child.parent = None
                return child
        # No matching child — start fresh from the current position
        return MCTSNode(game_state=game, piece=piece)

    @staticmethod
    def _boards_equal(a: np.ndarray, b: np.ndarray) -> bool:
        return bool((a == b).all())

    # ------------------------------------------------------------------ #
    #  Rollout                                                             #
    # ------------------------------------------------------------------ #

    def _rollout(self, state: Connect4Game) -> float:
        """
        Guided rollout: prefer immediately winning or blocking moves
        before falling back to a random choice.
        Returns 1.0 (agent wins), -1.0 (agent loses), or 0.0 (draw).
        """
        board          = state.board.copy()
        current_player = state.current_player
        opponent       = self.opponent_id if current_player == self.player_id else self.player_id
        rows, cols     = state.row_count, state.column_count

        while True:
            valid = [c for c in range(cols) if board[rows - 1][c] == 0]
            if not valid:
                return 0.0  # Draw

            # Prefer: win immediately > block opponent's win > random
            move = (
                self._find_winning_move(board, rows, cols, valid, current_player)
                or self._find_winning_move(board, rows, cols, valid, opponent)
                or random.choice(valid)
            )

            row = self._drop_piece(board, rows, move, current_player)
            if row == -1:
                return 0.0

            if self._check_winner_fast(board, rows, cols, row, move, current_player):
                return 1.0 if current_player == self.player_id else -1.0

            current_player, opponent = opponent, current_player

    # ------------------------------------------------------------------ #
    #  Backpropagation                                                     #
    # ------------------------------------------------------------------ #

    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        while node is not None:
            node.update(result)
            node = node.parent

    # ------------------------------------------------------------------ #
    #  Fast board helpers (operate on raw numpy arrays — no clone needed) #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _drop_piece(board: np.ndarray, rows: int, col: int, player: int) -> int:
        """Drop a piece and return the landed row, or -1 if column is full."""
        for row in range(rows):
            if board[row][col] == 0:
                board[row][col] = player
                return row
        return -1

    @staticmethod
    def _check_winner_fast(board: np.ndarray, rows: int, cols: int,
                            row: int, col: int, player: int) -> bool:
        """
        O(1) win check relative to board size — only examines the four
        directions through the last-placed piece.
        """
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < rows and 0 <= c < cols and board[r][c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def _find_winning_move(self, board: np.ndarray, rows: int, cols: int,
                           valid: list[int], player: int) -> int | None:
        """Return the first column that wins immediately for `player`, or None."""
        for col in valid:
            for row in range(rows):
                if board[row][col] == 0:
                    board[row][col] = player
                    won = self._check_winner_fast(board, rows, cols, row, col, player)
                    board[row][col] = 0
                    if won:
                        return col
                    break
        return None