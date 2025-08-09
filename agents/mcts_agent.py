from agents.agents_interface import AgentInterface
import numpy as np
import random
import copy
import math
from typing import List, Optional

class MCTSNode:
    def __init__(self, board: np.ndarray, player_id: int, parent: Optional['MCTSNode']=None, move: Optional[int]=None):
        self.board = board
        self.player_id = player_id
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.move = move  
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = self._get_valid_moves(board)

    def _get_valid_moves(self, board: np.ndarray) -> List[int]:
        rows, cols = board.shape
        top = rows - 1
        return [c for c in range(cols) if board[top][c] == 0]

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def best_child(self, c: float) -> 'MCTSNode':
        best = None
        best_score = -float('inf')
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploit = child.wins / child.visits
                explore = math.sqrt(math.log(self.visits) / child.visits)
                score = exploit + c * explore
            if score > best_score:
                best_score = score
                best = child
        assert best is not None
        return best

    def add_child(self, move: int, next_board: np.ndarray, next_player: int) -> 'MCTSNode':
        child = MCTSNode(next_board, next_player, parent=self, move=move)
        self.children.append(child)
        return child


class MCTSAgent(AgentInterface):
    def __init__(self, simulations: int = 1500, c: float = math.sqrt(2), seed: Optional[int] = None):
        self.simulations = simulations
        self.c = c
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def select_move(self, game) -> int:
        root = MCTSNode(copy.deepcopy(game.board), int(game.current_player))

        for _ in range(self.simulations):
            node = root

            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c)

            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                next_board = self._simulate_move(node.board, move, node.player_id)
                next_player = 2 if node.player_id == 1 else 1
                node = node.add_child(move, next_board, next_player)

            prev_player = 2 if node.player_id == 1 else 1
            if self._has_won(node.board, prev_player):
                result = prev_player
            else:

                result = self._simulate_random_game(copy.deepcopy(node.board), node.player_id)

            self._backpropagate(node, result)

        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return int(best_child.move)

        valid_moves = self._get_valid_moves_from_board(game.board)
        return int(random.choice(valid_moves)) if valid_moves else -1

    def _get_valid_moves_from_board(self, board: np.ndarray) -> List[int]:
        rows, cols = board.shape
        top = rows - 1
        return [c for c in range(cols) if board[top][c] == 0]

    def _simulate_move(self, board: np.ndarray, col: int, player_id: int) -> np.ndarray:
        new_board = copy.deepcopy(board)
        rows = new_board.shape[0]
        for r in range(rows): 
            if new_board[r][col] == 0:
                new_board[r][col] = player_id
                return new_board

        return new_board

    def _simulate_random_game(self, board: np.ndarray, player_to_move: int) -> int:
        current = player_to_move
        while True:
            moves = self._get_valid_moves_from_board(board)
            if not moves: 
                return 0
            col = random.choice(moves)
            board = self._simulate_move(board, col, current)
            if self._has_won(board, current):
                return current
            current = 2 if current == 1 else 1

    def _backpropagate(self, node: MCTSNode, winner: int) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            if winner == 0:
                cur.wins += 0.5
            elif winner == cur.player_id:
                cur.wins += 1.0
            # else add 0 for a loss
            cur = cur.parent

    # ---- Win check (board uses row 0 as bottom) ----
    def _has_won(self, board: np.ndarray, player_id: int) -> bool:
        ROWS, COLS = board.shape
        # horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if (board[r][c:c+4] == player_id).all():
                    return True
        # vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                if board[r][c] == player_id and board[r+1][c] == player_id and board[r+2][c] == player_id and board[r+3][c] == player_id:
                    return True
        # diag up-right (bottom-left to top-right)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if board[r][c] == player_id and board[r+1][c+1] == player_id and board[r+2][c+2] == player_id and board[r+3][c+3] == player_id:
                    return True
        # diag down-right (top-left to bottom-right in UI; here: from upper rows toward bottom)
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if board[r][c] == player_id and board[r-1][c+1] == player_id and board[r-2][c+2] == player_id and board[r-3][c+3] == player_id:
                    return True
        return False
