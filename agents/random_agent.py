import random

class RandomAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        
    def get_valid_moves(self, game):
        valid_moves = []
        for col in range(game.column_count):
            if game.is_valid_location(game.board, col):
                valid_moves.append(col)
        return valid_moves
    
    def select_move(self, game):
        valid_moves = self.get_valid_moves(game)
        if not valid_moves:
            return -1
        return random.choice(valid_moves)
    