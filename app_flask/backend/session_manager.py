import uuid
import time
from typing import Optional, Dict, Any
from game.connect4 import Connect4Game


class GameSession:
    def __init__(
        self,
        game_id: str,
        mode: str,
        player1_type: str,
        player2_type: str,
        player1_config: dict,
        player2_config: dict,
        agent1=None,
        agent2=None,
    ):
        self.game_id = game_id
        self.mode = mode
        self.player1_type = player1_type
        self.player2_type = player2_type
        self.player1_config = player1_config
        self.player2_config = player2_config
        self.agent1 = agent1
        self.agent2 = agent2

        self.game = Connect4Game()
        self.move_history: list = []
        self.status = "active"  # "active", "finished"
        self.winner: Optional[str] = None  # "player1", "player2", "draw"
        self.started_at = time.time()
        self.finished_at: Optional[float] = None

    def board_as_list(self):
        import numpy as np
        return self.game.board.tolist()

    def to_state(self) -> dict:
        return {
            "game_id": self.game_id,
            "mode": self.mode,
            "current_player": self.game.current_player,
            "board": self.board_as_list(),
            "status": self.status,
            "winner": self.winner,
            "move_history": self.move_history,
            "player1_type": self.player1_type,
            "player2_type": self.player2_type,
        }


# In-memory store
_sessions: Dict[str, GameSession] = {}


def create_session(
    mode: str,
    player1_type: str,
    player2_type: str,
    player1_config: dict,
    player2_config: dict,
) -> GameSession:
    from agent_factory import create_agent

    game_id = str(uuid.uuid4())
    agent1 = None
    agent2 = None

    if player1_type != "human":
        agent1 = create_agent(player1_type, 1, player1_config)
    if player2_type != "human":
        agent2 = create_agent(player2_type, 2, player2_config)

    session = GameSession(
        game_id=game_id,
        mode=mode,
        player1_type=player1_type,
        player2_type=player2_type,
        player1_config=player1_config,
        player2_config=player2_config,
        agent1=agent1,
        agent2=agent2,
    )
    _sessions[game_id] = session
    return session


def get_session(game_id: str) -> Optional[GameSession]:
    return _sessions.get(game_id)


def delete_session(game_id: str):
    _sessions.pop(game_id, None)
