import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from session_manager import create_session, get_session, delete_session
from database import get_db, GameRecord
from agent_factory import AGENT_REGISTRY

router = APIRouter(prefix="/api", tags=["game"])


# ── Schemas ──────────────────────────────────────────────────────────────── #

class NewGameRequest(BaseModel):
    mode: str  # "hvh", "hva", "ava"
    player1_type: str
    player2_type: str
    player1_config: dict = {}
    player2_config: dict = {}


class MoveRequest(BaseModel):
    column: int


# ── Helpers ──────────────────────────────────────────────────────────────── #

def _apply_move(session, col: int) -> dict:
    game = session.game
    player = game.current_player

    if not game.is_valid_location(game.board, col):
        raise HTTPException(status_code=400, detail="Invalid column")

    success, row = game.make_move(col)
    if not success:
        raise HTTPException(status_code=400, detail="Column is full")

    session.move_history.append({"player": player, "col": col, "row": row})

    # Check win / draw
    if game.check_winner(row, col, player):
        session.status = "finished"
        session.winner = "player1" if player == 1 else "player2"
        session.finished_at = time.time()
    elif game.check_draw():
        session.status = "finished"
        session.winner = "draw"
        session.finished_at = time.time()
    else:
        game.switch_player()

    return session.to_state()


def _get_agent_move(session) -> int:
    game = session.game
    agent = session.agent1 if game.current_player == 1 else session.agent2
    if agent is None:
        raise HTTPException(status_code=400, detail="No agent for current player")
    return agent.select_move(game)


def _save_game(session, db: Session):
    duration = None
    if session.finished_at:
        duration = session.finished_at - session.started_at

    record = GameRecord(
        mode=session.mode,
        player1_type=session.player1_type,
        player2_type=session.player2_type,
        player1_config=session.player1_config,
        player2_config=session.player2_config,
        winner=session.winner,
        total_moves=len(session.move_history),
        duration_seconds=duration,
        final_board=session.board_as_list(),
        move_history=session.move_history,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ── Routes ───────────────────────────────────────────────────────────────── #

@router.get("/agents")
def list_agents():
    return AGENT_REGISTRY


@router.post("/games")
def new_game(req: NewGameRequest, db: Session = Depends(get_db)):
    mode = req.mode
    p1, p2 = req.player1_type, req.player2_type

    # Validate mode
    if mode == "hvh" and (p1 != "human" or p2 != "human"):
        raise HTTPException(400, "HvH requires both players to be human")
    if mode == "hva" and not (p1 == "human" or p2 == "human"):
        raise HTTPException(400, "HvA requires exactly one human")
    if mode == "ava" and (p1 == "human" or p2 == "human"):
        raise HTTPException(400, "AvA requires both players to be agents")

    session = create_session(mode, p1, p2, req.player1_config, req.player2_config)
    state = session.to_state()

    # If player 1 is an agent, make its move immediately
    if p1 != "human" and session.status == "active":
        col = _get_agent_move(session)
        state = _apply_move(session, col)

        if session.status == "finished":
            _save_game(session, db)

    return state


@router.get("/games/{game_id}")
def get_game(game_id: str):
    session = get_session(game_id)
    if not session:
        raise HTTPException(404, "Game not found")
    return session.to_state()


@router.post("/games/{game_id}/move")
def make_human_move(game_id: str, req: MoveRequest, db: Session = Depends(get_db)):
    session = get_session(game_id)
    if not session:
        raise HTTPException(404, "Game not found")
    if session.status != "active":
        raise HTTPException(400, "Game is already finished")

    # Validate it's a human turn
    current = session.game.current_player
    is_human_turn = (
        (current == 1 and session.player1_type == "human") or
        (current == 2 and session.player2_type == "human")
    )
    if not is_human_turn:
        raise HTTPException(400, "Not human's turn")

    state = _apply_move(session, req.column)

    # After human move, let agent respond if game still active
    if session.status == "active":
        next_player = session.game.current_player
        next_is_agent = (
            (next_player == 1 and session.player1_type != "human") or
            (next_player == 2 and session.player2_type != "human")
        )
        if next_is_agent:
            col = _get_agent_move(session)
            state = _apply_move(session, col)

    if session.status == "finished":
        _save_game(session, db)

    return state


@router.post("/games/{game_id}/agent-move")
def make_agent_move(game_id: str, db: Session = Depends(get_db)):
    """Used for AvA mode - steps one agent move at a time."""
    session = get_session(game_id)
    if not session:
        raise HTTPException(404, "Game not found")
    if session.status != "active":
        raise HTTPException(400, "Game is already finished")

    col = _get_agent_move(session)
    state = _apply_move(session, col)

    if session.status == "finished":
        _save_game(session, db)

    return state


@router.post("/games/{game_id}/auto-play")
def auto_play_ava(game_id: str, db: Session = Depends(get_db)):
    """Run entire AvA game to completion, return all states."""
    session = get_session(game_id)
    if not session:
        raise HTTPException(404, "Game not found")
    if session.mode != "ava":
        raise HTTPException(400, "Only for AvA games")

    states = []
    while session.status == "active":
        col = _get_agent_move(session)
        state = _apply_move(session, col)
        states.append(state)
        if len(states) > 50:  # safety guard
            break

    if session.status == "finished":
        _save_game(session, db)

    return {"states": states, "final": session.to_state()}


@router.delete("/games/{game_id}")
def abandon_game(game_id: str):
    delete_session(game_id)
    return {"ok": True}


# ── History ──────────────────────────────────────────────────────────────── #

@router.get("/history")
def get_history(limit: int = 20, db: Session = Depends(get_db)):
    records = (
        db.query(GameRecord)
        .order_by(GameRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "mode": r.mode,
            "player1_type": r.player1_type,
            "player2_type": r.player2_type,
            "player1_config": r.player1_config,
            "player2_config": r.player2_config,
            "winner": r.winner,
            "total_moves": r.total_moves,
            "duration_seconds": r.duration_seconds,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]


@router.get("/history/{record_id}")
def get_history_detail(record_id: int, db: Session = Depends(get_db)):
    record = db.query(GameRecord).filter(GameRecord.id == record_id).first()
    if not record:
        raise HTTPException(404, "Record not found")
    return {
        "id": record.id,
        "mode": record.mode,
        "player1_type": record.player1_type,
        "player2_type": record.player2_type,
        "player1_config": record.player1_config,
        "player2_config": record.player2_config,
        "winner": record.winner,
        "total_moves": record.total_moves,
        "duration_seconds": record.duration_seconds,
        "final_board": record.final_board,
        "move_history": record.move_history,
        "created_at": record.created_at.isoformat(),
    }


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func
    total = db.query(func.count(GameRecord.id)).scalar()
    by_mode = (
        db.query(GameRecord.mode, func.count(GameRecord.id))
        .group_by(GameRecord.mode)
        .all()
    )
    by_winner = (
        db.query(GameRecord.winner, func.count(GameRecord.id))
        .group_by(GameRecord.winner)
        .all()
    )
    return {
        "total_games": total,
        "by_mode": {m: c for m, c in by_mode},
        "by_winner": {w: c for w, c in by_winner},
    }
