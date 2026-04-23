import json
import re
import os
import numpy as np
import urllib.request
from agents.agents_interface import AgentInterface

# ── Ollama config ────────────────────────────────────────────────────── #
# Make sure Ollama is running:  ollama serve
# Pull a model first:           ollama pull gemma3  (or llama3.2, mistral, etc.)

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")          # swap to any model you have pulled

# ── Prompt ───────────────────────────────────────────────────────────── #

SYSTEM_PROMPT = """\
You are an expert Connect 4 AI. Given a board state, pick the best column.

Rules:
- Board is 6 rows x 7 cols. Row 0 = bottom, row 5 = top.
- 0=empty, 1=Player1, 2=Player2.
- Columns 0-6 (left to right).

Priorities:
1. Win immediately if possible.
2. Block opponent's immediate win.
3. Build or extend your own threats (3-in-a-row with open end).
4. Block opponent double-threats.
5. Prefer centre columns (3 > 2,4 > 1,5 > 0,6).
6. Never give opponent a free winning square above your piece.

Respond ONLY with a JSON object, nothing else:
{"col": <0-6>, "reason": "<one short sentence>"}\
"""

def _board_text(board: np.ndarray) -> str:
    lines = []
    for row in range(5, -1, -1):
        lines.append("row %d: %s" % (row, " ".join(str(int(board[row][c])) for c in range(7))))
    lines.append("col:   0 1 2 3 4 5 6")
    return "\n".join(lines)

def _build_prompt(board: np.ndarray, player_id: int, valid_moves: list) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"You are Player {player_id}. Opponent is Player {3 - player_id}.\n\n"
        f"Board:\n{_board_text(board)}\n\n"
        f"Valid columns: {valid_moves}\n\n"
        'Respond with JSON only: {"col": <int>, "reason": "<str>"}'
    )

# ── Local heuristics (fast fallback + forced-move detection) ─────────── #

def _score_window(window: list, piece: int) -> int:
    opp = 2 if piece == 1 else 1
    if window.count(piece) == 4:                               return 100
    if window.count(piece) == 3 and window.count(0) == 1:     return 5
    if window.count(piece) == 2 and window.count(0) == 2:     return 2
    if window.count(opp)   == 3 and window.count(0) == 1:     return -4
    return 0

def _heuristic(board: np.ndarray, piece: int, rows=6, cols=7) -> int:
    score = [int(board[r][cols // 2]) for r in range(rows)].count(piece) * 3
    for r in range(rows):
        for c in range(cols - 3):
            score += _score_window([int(board[r][c+i]) for i in range(4)], piece)
    for c in range(cols):
        for r in range(rows - 3):
            score += _score_window([int(board[r+i][c]) for i in range(4)], piece)
    for r in range(rows - 3):
        for c in range(cols - 3):
            score += _score_window([int(board[r+i][c+i]) for i in range(4)], piece)
    for r in range(3, rows):
        for c in range(cols - 3):
            score += _score_window([int(board[r-i][c+i]) for i in range(4)], piece)
    return score

def _drop(board, col, piece, rows=6):
    t = board.copy()
    for r in range(rows):
        if t[r][col] == 0:
            t[r][col] = piece
            return t
    return t

def _has_four(board, piece, rows=6, cols=7):
    for r in range(rows):
        for c in range(cols - 3):
            if all(board[r][c+i] == piece for i in range(4)): return True
    for c in range(cols):
        for r in range(rows - 3):
            if all(board[r+i][c] == piece for i in range(4)): return True
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r+i][c+i] == piece for i in range(4)): return True
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(board[r-i][c+i] == piece for i in range(4)): return True
    return False

def _forced_move(board, valid_moves, player_id):
    """Return winning col, then blocking col, or None."""
    opp = 2 if player_id == 1 else 1
    for piece in (player_id, opp):
        for col in valid_moves:
            if _has_four(_drop(board, col, piece), piece):
                return col
    return None

# ── Ollama call ──────────────────────────────────────────────────────── #

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 15) -> str:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic → fastest
            "num_predict": 60,    # tiny reply → low latency
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data.get("response", "")

def _parse_col(text: str) -> int | None:
    # Try strict JSON first
    try:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            return int(json.loads(m.group())["col"])
    except Exception:
        pass
    # Fallback: first digit in 0-6 range
    m = re.search(r'\b([0-6])\b', text)
    return int(m.group(1)) if m else None

# ── Agent ────────────────────────────────────────────────────────────── #

class LLMAgent(AgentInterface):
    """
    Connect 4 agent backed by a local Ollama model.

    Speed design
    ------------
    * Forced win / block resolved locally — no LLM call at all.
    * num_predict=60 keeps token generation very short.
    * temperature=0 → deterministic, no sampling overhead.
    * Pure stdlib urllib — zero extra dependencies.
    * Falls back to a heuristic scorer if Ollama is unavailable or
      returns an illegal column.

    Quick setup
    -----------
        ollama serve                   # start the daemon
        ollama pull gemma3             # recommended: fast & smart
        # alternatives: llama3.2, mistral, phi3, qwen2.5:0.5b (ultra-light)

    To change model, edit OLLAMA_MODEL at the top of this file,
    or pass model="<name>" to the constructor.
    """

    def __init__(self, player_id: int, model: str = OLLAMA_MODEL, verbose: bool = False):
        super().__init__(player_id)
        self.model   = model
        self.verbose = verbose

    def select_move(self, game) -> int:
        board       = game.get_board()
        valid_moves = game.get_valid_locations()

        # 1. Instant forced move — win or block, no LLM needed
        forced = _forced_move(board, valid_moves, self.player_id)
        if forced is not None:
            if self.verbose:
                print(f"[LLMAgent P{self.player_id}] forced col={forced}")
            return forced

        # 2. Ask Ollama
        col = self._ask_ollama(board, valid_moves)

        # 3. Validate; fall back to heuristic if bad/missing response
        if col not in valid_moves:
            col = self._best_heuristic(board, valid_moves)
            if self.verbose:
                print(f"[LLMAgent P{self.player_id}] heuristic fallback col={col}")
        elif self.verbose:
            print(f"[LLMAgent P{self.player_id}] LLM chose col={col}")

        return col

    # ── internals ──────────────────────────────────────────────────── #

    def _ask_ollama(self, board, valid_moves):
        try:
            prompt = _build_prompt(board, self.player_id, valid_moves)
            raw    = _call_ollama(prompt, model=self.model)
            if self.verbose:
                print(f"[LLMAgent P{self.player_id}] ollama: {raw!r}")
            return _parse_col(raw)
        except Exception as exc:
            if self.verbose:
                print(f"[LLMAgent P{self.player_id}] ollama error: {exc}")
            return None

    def _best_heuristic(self, board, valid_moves):
        return max(
            valid_moves,
            key=lambda c: _heuristic(_drop(board, c, self.player_id), self.player_id),
        )