from agents.agents_interface import AgentInterface
import numpy as np
import random
import copy

class MinMaxAgent(AgentInterface):
    def __init__(self, player_id, depth=4):
        super().__init__(player_id)
        self.depth = depth
        self.opponent_id = 1 if player_id == 2 else 2

    def _get_valid_moves_from_board(self, board):
        valid_moves = []
        for col in range(board.shape[1]):
            if board[0][col] == 0:
                valid_moves.append(col)
        return valid_moves

    def _simulate_move(self, board, col, player_id):
        new_board = copy.deepcopy(board)
        for row in reversed(range(new_board.shape[0])):
            if new_board[row][col] == 0:
                new_board[row][col] = player_id
                break
        return new_board

    def _has_won(self, board, player_id):
        ROWS, COLS = board.shape
        for row in range(ROWS):
            for col in range(COLS):
                if board[row][col] != player_id:
                    continue
                # Horizontal
                if col <= COLS - 4 and all(board[row][col + i] == player_id for i in range(4)):
                    return True
                # Vertical
                if row <= ROWS - 4 and all(board[row + i][col] == player_id for i in range(4)):
                    return True
                # Diagonal /
                if row >= 3 and col <= COLS - 4 and all(board[row - i][col + i] == player_id for i in range(4)):
                    return True
                # Diagonal \
                if row <= ROWS - 4 and col <= COLS - 4 and all(board[row + i][col + i] == player_id for i in range(4)):
                    return True
        return False

    def _is_terminal_node(self, board):
        return (
            self._has_won(board, self.player_id) or
            self._has_won(board, self.opponent_id) or
            len(self._get_valid_moves_from_board(board)) == 0
        )

    def evaluate_board(self, board):
        score = 0
        ROWS, COLS = board.shape
        center_col = COLS // 2

        center_array = [int(board[row][center_col]) for row in range(ROWS)]
        center_count = center_array.count(self.player_id)
        score += center_count * 3

        def evaluate_window(window):
            s = 0
            if window.count(self.player_id) == 4:
                s += 100
            elif window.count(self.player_id) == 3 and window.count(0) == 1:
                s += 10
            elif window.count(self.player_id) == 2 and window.count(0) == 2:
                s += 5
            if window.count(self.opponent_id) == 3 and window.count(0) == 1:
                s -= 80
            return s

        # Horizontal
        for row in range(ROWS):
            for col in range(COLS - 3):
                window = list(board[row, col:col + 4])
                score += evaluate_window(window)

        # Vertical
        for col in range(COLS):
            for row in range(ROWS - 3):
                window = list(board[row:row + 4, col])
                score += evaluate_window(window)

        # Positive diagonal
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                window = [board[row + i][col + i] for i in range(4)]
                score += evaluate_window(window)

        # Negative diagonal
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                window = [board[row - i][col + i] for i in range(4)]
                score += evaluate_window(window)

        return score

    def _minmax(self, board, depth, maximizing_player):
        valid_locations = self._get_valid_moves_from_board(board)
        is_terminal = self._is_terminal_node(board)

        if depth == 0 or is_terminal:
            if is_terminal:
                if self._has_won(board, self.player_id):
                    return float('inf'), None
                elif self._has_won(board, self.opponent_id):
                    return float('-inf'), None
                else:
                    return 0, None
            else:
                return self.evaluate_board(board), None

        if maximizing_player:
            value = float('-inf')
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                new_board = self._simulate_move(board, col, self.player_id)
                new_score, _ = self._minmax(new_board, depth - 1, False)
                if new_score > value:
                    value = new_score
                    best_col = col
            return value, best_col
        else:
            value = float('inf')
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                new_board = self._simulate_move(board, col, self.opponent_id)
                new_score, _ = self._minmax(new_board, depth - 1, True)
                if new_score < value:
                    value = new_score
                    best_col = col
            return value, best_col

    def select_move(self, game):
        valid_moves = super().get_valid_moves(game)
        best_score = float('-inf')
        best_col = random.choice(valid_moves)

        for col in valid_moves:
            board_copy = self._simulate_move(game.board, col, self.player_id)
            score, _ = self._minmax(board_copy, self.depth - 1, False)
            if score > best_score:
                best_score = score
                best_col = col

        return best_col
