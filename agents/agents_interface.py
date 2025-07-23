from abc import ABC, abstractmethod

class AgentInterface(ABC):
    def __init__(self, player_id):
        self.player_id = player_id
        
    def get_valid_moves(self, game):
        valid_moves = []
        for col in range(game.column_count):
            if game.is_valid_location(game.board, col):
                valid_moves.append(col)
        return valid_moves
        
    @abstractmethod
    def select_move(self, game):
        pass
    
    