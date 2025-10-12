from agents.agents_interface import AgentInterface
import copy
import numpy as np

class RuleBasedAgent(AgentInterface):
    def __init__(self, player_id):
        super().__init__(player_id)

    def _get_next_open_row(self, board, col):
        for row in reversed(range(board.shape[0])):
            if board[row][col] == 0:
                return row
        return None

    def _check_win(self, board, row, col, player_id):
        ROWS, COLS = board.shape

        def count_dir(delta_row, delta_col):
            count = 0
            r, c = row + delta_row, col + delta_col
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player_id:
                count += 1
                r += delta_row
                c += delta_col
            return count

        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1 + count_dir(dr, dc) + count_dir(-dr, -dc)
            if count >= 4:
                return True
        return False

    def _can_win_next(self, game, player_id):
        board_state = copy.deepcopy(game.board)
        valid_moves = self.get_valid_moves(game)
        for col in valid_moves:
            tmp_board = copy.deepcopy(board_state)
            row = self._get_next_open_row(tmp_board, col)
            if row is not None:
                tmp_board[row][col] = player_id
                if self._check_win(tmp_board, row, col, player_id):
                    return col
        return None

    def _must_block(self, game, player_id):
        opponent_id = 1 if player_id == 2 else 2
        return self._can_win_next(game, opponent_id)

    def _prefer_center(self, game):
        center = game.column_count // 2
        if center in self.get_valid_moves(game):
            return center
        return None

    def _double_threat(self, game, player_id):
        # Attempt to create a situation where there are two ways to win on the next turn
        valid_moves = self.get_valid_moves(game)
        board_state = copy.deepcopy(game.board)
        for col in valid_moves:
            tmp_board = copy.deepcopy(board_state)
            row = self._get_next_open_row(tmp_board, col)
            if row is not None:
                tmp_board[row][col] = player_id
                win_count = 0
                # For all valid next moves, check if the agent can win in two places next turn
                for next_col in valid_moves:
                    if next_col != col:
                        next_board = copy.deepcopy(tmp_board)
                        next_row = self._get_next_open_row(next_board, next_col)
                        if next_row is not None:
                            next_board[next_row][next_col] = player_id
                            if self._check_win(next_board, next_row, next_col, player_id):
                                win_count += 1
                    if win_count >= 2:
                        return col
        return None

    def get_valid_moves(self, game):
        valid_moves = []
        for col in range(game.column_count):
            if game.is_valid_location(game.board, col):
                valid_moves.append(col)
        return valid_moves

    def select_move(self, game):
        # 1. Win if possible
        win_col = self._can_win_next(game, self.player_id)
        if win_col is not None:
            return win_col

        # 2. Block if necessary
        block_col = self._must_block(game, self.player_id)
        if block_col is not None:
            return block_col

        # 3. Double threat (set up two-in-a-row threats)
        double_col = self._double_threat(game, self.player_id)
        if double_col is not None:
            return double_col

        # 4. Prefer center column
        center_col = self._prefer_center(game)
        if center_col is not None:
            return center_col

        # 5. Prefer columns next to center
        center = game.column_count // 2
        near_center_moves = sorted(self.get_valid_moves(game), key=lambda c: abs(c - center))
        if near_center_moves:
            return near_center_moves[0]

        # 6. Fallback: pick any valid move at random
        valid_moves = self.get_valid_moves(game)
        if valid_moves:
            return np.random.choice(valid_moves)

        # 7. No moves available
        return -1
