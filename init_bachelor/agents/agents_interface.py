from abc import ABC, abstractmethod

class AgentInterface(ABC):
    def __init__(self, player_id):
        self.player_id = player_id
        self.opponent_id = 2 if player_id == 1 else 1
        
    def get_valid_moves(self, game):
        valid_moves = []
        for col in range(game.column_count):
            if game.is_valid_location(game.board, col):
                valid_moves.append(col)
        return valid_moves
        
    @abstractmethod
    def select_move(self, game):
        pass
    
    