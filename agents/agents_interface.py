from abc import ABC, abstractmethod

class AgentInterface(ABC):
    def __init__(self, player_id):
        self.player_id = player_id
        
    @abstractmethod
    def get_valid_moves(self, game):
        pass
        
    @abstractmethod
    def select_move(self, game):
        pass
    
    