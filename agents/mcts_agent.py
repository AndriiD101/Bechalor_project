from agents.agents_interface import AgentInterface
from agents.minmax_agent import MinMaxAgent
import numpy as np
import random
import copy

class MCTSNode:
    def __init__(self, board, player_id, parent=None, move=None):
        self.board = board
        self.player_id = player_id
        self.parent = parent
        self.children = []
        self.move = move
        self.visits = 0
        self.wins = 0 
        self.untried_moves = self._get_valid_moves(board)

    
    def _get_valid_moves(self, board):
        valid_moves = []
        for col in range(board.shape[1]):
            if board[0][col] == 0:
                valid_moves.append(col)
        return valid_moves
    
class MCTSAgent(AgentInterface, MinMaxAgent):
    def __init__(self, player_id, simulation=1000):
        super().__init__(player_id)
        self.opponent_id = 1 if player_id == 2 else 2

        self.simulation = simulation
    
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
    
    def _simulate_random_game(self, board, current_player):
        board_copy = copy.deepcopy(board)
        player = current_player
        
        while True:
            valid_moves = self._get_valid_moves_from_board(board_copy)
            if not valid_moves:
                return 0
            
            move = random.choice(valid_moves)
            board_copy = self._simulate_move(board_copy, move, player)
            
            if self._has_won(board_copy, player):
                return player

            player = 1 if player == 2 else 2
    
    def _expand(self, node):
        move = node.untried_moves.pop()  # Remove one untried move
        new_board = self._simulate_move(node.board, move, node.player_id)
        next_player = 1 if node.player_id == 2 else 2
        child_node = MCTSNode(
            board=new_board,
            player_id=next_player,
            parent=node,
            move=move
        )
        node.children.append(child_node)
        return child_node

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            if result == self.player_id:
                node.wins += 1
            elif result == 0:
                node.wins += 0.5
            node = node.parent
    
    def _select_best_child(self, node):
        best_child = None
        best_value = float('-inf')
        for child in node.children:
            if child.visits == 0:
                continue
            ucb1 = (child.wins / child.visits) + np.sqrt(2 * np.log(node.visits) / child.visits)
            if ucb1 > best_value:
                best_value = ucb1
                best_child = child
        return best_child

    
    def select_move(self, game):
        root = MCTSNode(copy.deepcopy(game.board), self.player_id)

        for _ in range(self.simulations):
            node = root

            # Selection
            while node.untried_moves == [] and node.children:
                node = self._select_best_child(node)

            # Expansion
            if node.untried_moves:
                node = self._expand(node)

            # Simulation
            result = self._simulate_random_game(copy.deepcopy(node.board), node.player_id)

            # Backpropagation
            self._backpropagate(node, result)

        # Choose the most visited move
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.move
        else:
            valid_moves = self._get_valid_moves_from_board(game.board)
            return random.choice(valid_moves) if valid_moves else -