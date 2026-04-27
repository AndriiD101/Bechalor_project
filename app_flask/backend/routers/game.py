import time
import os
import shutil
from flask import Blueprint, request, jsonify, send_file, g
from sqlalchemy.orm import Session

from session_manager import create_session, get_session, delete_session
from database import SessionLocal, GameRecord
from agent_factory import AGENT_REGISTRY
from gif_generator import generate_connect4_gif

game_bp = Blueprint("game", __name__, url_prefix="/api")

os.makedirs("uploaded_models", exist_ok=True)


# ── DB dependency ─────────────────────────────────────────────────────────── #

def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


# ── Helpers ──────────────────────────────────────────────────────────────── #

def _apply_move(session, col: int) -> dict:
    game = session.game
    player = game.current_player

    if not game.is_valid_location(game.board, col):
        return None, "Invalid column"

    success, row = game.make_move(col)
    if not success:
        return None, "Column is full"

    session.move_history.append({"player": player, "col": col, "row": row})

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

    return session.to_state(), None


def _get_agent_move(session) -> int:
    game = session.game
    agent = session.agent1 if game.current_player == 1 else session.agent2
    if agent is None:
        return None, "No agent for current player"
    return agent.select_move(game), None


def _save_game(session, db):
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
        tournament_id=getattr(session, "tournament_id", None),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ── Routes ───────────────────────────────────────────────────────────────── #

@game_bp.get("/agents")
def list_agents():
    return jsonify(AGENT_REGISTRY)


@game_bp.post("/games")
def new_game():
    data = request.get_json(force=True)
    mode = data.get("mode")
    p1 = data.get("player1_type")
    p2 = data.get("player2_type")
    p1_config = data.get("player1_config", {})
    p2_config = data.get("player2_config", {})
    tournament_id = data.get("tournament_id")

    if mode == "hvh" and (p1 != "human" or p2 != "human"):
        return jsonify({"detail": "HvH requires both players to be human"}), 400
    if mode == "hva" and not (p1 == "human" or p2 == "human"):
        return jsonify({"detail": "HvA requires exactly one human"}), 400
    if mode == "ava" and (p1 == "human" or p2 == "human"):
        return jsonify({"detail": "AvA requires both players to be agents"}), 400

    session = create_session(mode, p1, p2, p1_config, p2_config)
    session.tournament_id = tournament_id

    state = session.to_state()

    if p1 != "human" and session.status == "active":
        col, err = _get_agent_move(session)
        if err:
            return jsonify({"detail": err}), 400
        state, err = _apply_move(session, col)
        if err:
            return jsonify({"detail": err}), 400

        if session.status == "finished":
            db = get_db()
            try:
                _save_game(session, db)
            finally:
                db.close()

    return jsonify(state)


@game_bp.get("/games/<game_id>")
def get_game(game_id):
    session = get_session(game_id)
    if not session:
        return jsonify({"detail": "Game not found"}), 404
    return jsonify(session.to_state())


@game_bp.post("/games/<game_id>/move")
def make_human_move(game_id):
    session = get_session(game_id)
    if not session:
        return jsonify({"detail": "Game not found"}), 404
    if session.status != "active":
        return jsonify({"detail": "Game is already finished"}), 400

    data = request.get_json(force=True)
    col = data.get("column")

    current = session.game.current_player
    is_human_turn = (
        (current == 1 and session.player1_type == "human") or
        (current == 2 and session.player2_type == "human")
    )
    if not is_human_turn:
        return jsonify({"detail": "Not human's turn"}), 400

    state, err = _apply_move(session, col)
    if err:
        return jsonify({"detail": err}), 400

    if session.status == "active":
        next_player = session.game.current_player
        next_is_agent = (
            (next_player == 1 and session.player1_type != "human") or
            (next_player == 2 and session.player2_type != "human")
        )
        if next_is_agent:
            col, err = _get_agent_move(session)
            if err:
                return jsonify({"detail": err}), 400
            state, err = _apply_move(session, col)
            if err:
                return jsonify({"detail": err}), 400

    if session.status == "finished":
        db = get_db()
        try:
            _save_game(session, db)
        finally:
            db.close()

    return jsonify(state)


@game_bp.post("/games/<game_id>/agent-move")
def make_agent_move(game_id):
    session = get_session(game_id)
    if not session:
        return jsonify({"detail": "Game not found"}), 404
    if session.status != "active":
        return jsonify({"detail": "Game is already finished"}), 400

    col, err = _get_agent_move(session)
    if err:
        return jsonify({"detail": err}), 400
    state, err = _apply_move(session, col)
    if err:
        return jsonify({"detail": err}), 400

    if session.status == "finished":
        db = get_db()
        try:
            _save_game(session, db)
        finally:
            db.close()

    return jsonify(state)


@game_bp.post("/games/<game_id>/auto-play")
def auto_play_ava(game_id):
    session = get_session(game_id)
    if not session:
        return jsonify({"detail": "Game not found"}), 404
    if session.mode != "ava":
        return jsonify({"detail": "Only for AvA games"}), 400

    states = []
    while session.status == "active":
        col, err = _get_agent_move(session)
        if err:
            return jsonify({"detail": err}), 400
        state, err = _apply_move(session, col)
        if err:
            return jsonify({"detail": err}), 400
        states.append(state)
        if len(states) > 50:
            break

    if session.status == "finished":
        db = get_db()
        try:
            _save_game(session, db)
        finally:
            db.close()

    return jsonify({"states": states, "final": session.to_state()})


@game_bp.delete("/games/<game_id>")
def abandon_game(game_id):
    delete_session(game_id)
    return jsonify({"ok": True})


# ── History ──────────────────────────────────────────────────────────────── #

@game_bp.get("/history")
def get_history():
    limit = request.args.get("limit", 20, type=int)
    db = get_db()
    try:
        records = (
            db.query(GameRecord)
            .order_by(GameRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return jsonify([
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
                "tournament_id": r.tournament_id,
            }
            for r in records
        ])
    finally:
        db.close()


@game_bp.get("/history/<int:record_id>")
def get_history_detail(record_id):
    db = get_db()
    try:
        record = db.query(GameRecord).filter(GameRecord.id == record_id).first()
        if not record:
            return jsonify({"detail": "Record not found"}), 404
        return jsonify({
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
            "tournament_id": record.tournament_id,
        })
    finally:
        db.close()


@game_bp.get("/history/<int:record_id>/gif")
def get_game_gif(record_id):
    db = get_db()
    try:
        record = db.query(GameRecord).filter(GameRecord.id == record_id).first()
        if not record:
            return jsonify({"detail": "Record not found"}), 404
        if not record.move_history:
            return jsonify({"detail": "No moves recorded for this game"}), 400

        def get_pretty_name(agent_type):
            if agent_type == "human":
                return "Human"
            agent_info = AGENT_REGISTRY.get(agent_type)
            if agent_info:
                return agent_info.get("label", agent_type)
            return agent_type

        p1_label = get_pretty_name(record.player1_type)
        p2_label = get_pretty_name(record.player2_type)

        gif_dir = "saved_gifs"
        os.makedirs(gif_dir, exist_ok=True)
        filepath = os.path.join(gif_dir, f"game_{record_id}.gif")

        if not os.path.exists(filepath):
            generate_connect4_gif(
                record.move_history,
                filepath,
                p1_name=p1_label,
                p2_name=p2_label,
            )

        safe_p1 = p1_label.replace(" ", "_")
        safe_p2 = p2_label.replace(" ", "_")
        download_name = f"{safe_p1}_vs_{safe_p2}_game_{record_id}.gif"

        return send_file(filepath, mimetype="image/gif",
                         as_attachment=True, download_name=download_name)
    finally:
        db.close()


@game_bp.get("/stats")
def get_stats():
    from sqlalchemy import func
    db = get_db()
    try:
        total = db.query(func.count(GameRecord.id)).scalar()
        by_mode = db.query(GameRecord.mode, func.count(GameRecord.id)).group_by(GameRecord.mode).all()
        by_winner = db.query(GameRecord.winner, func.count(GameRecord.id)).group_by(GameRecord.winner).all()
        return jsonify({
            "total_games": total,
            "by_mode": {m: c for m, c in by_mode},
            "by_winner": {w: c for w, c in by_winner},
        })
    finally:
        db.close()


@game_bp.post("/upload-model")
def upload_model():
    if "file" not in request.files:
        return jsonify({"detail": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename.endswith((".pt", ".pth")):
        return jsonify({"detail": "Only PyTorch (.pt, .pth) files are allowed"}), 400

    file_path = os.path.join("uploaded_models", file.filename)
    file.save(file_path)
    return jsonify({"file_path": file_path})
