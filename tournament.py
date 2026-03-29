"""
Connect-4 Parameter Sweep Tournament
=====================================
Every agent variant (param combo) plays against every OTHER agent variant.
Configure PARAM_SWEEP below — each agent lists the values to try for each
parameter.  The script generates all combinations via itertools.product,
then runs every variant-vs-variant matchup.

HOW TO CONFIGURE
----------------
Edit PARAM_SWEEP: agent_name → {param: [values_to_try]}.
Agents with no params (Random, RuleBased) need no entry — they are added
automatically as single fixed variants.

HOW TO RUN
----------
    python tournament.py

Output
------
    sweep_results/<P1_label>_vs_<P2_label>.txt   — per-matchup file
    sweep_results/summary.csv                     — all results in one table
    (console) best param combo per agent at the end
"""

import csv
import itertools
import os
import time
from collections import defaultdict

from game import connect4
from agents import (
    mcts_agent,
    minmax_agent,
    rule_based_agent,
    random_agent,
    alfabetapruning_agent,
)
from agents.reinforcement_agent import DQNAgent


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← only edit this section
# ══════════════════════════════════════════════════════════════════════

PARAM_SWEEP = {
    # ── No-param agents (single fixed variant each) ───────────────────
    "Random":    {},
    "RuleBased": {},
    # ── Tree-search agents ────────────────────────────────────────────
    "MinMax": {
        "max_depth": [1, 2, 3, 4, 5, 6],
    },
    "AlphaBeta": {
        "max_depth": [1, 2, 3, 4, 5, 6],
    },
    "MCTS": {
        "max_iterations": [5000, 10000, 15000, 20000]
    },
    # ── DQN ───────────────────────────────────────────────────────────
    "DQN": {
        "model_path": ["checkpoints/dqn_final.pth"],
    },
}

GAMES_PER_MATCHUP = 50    # games per ordered (P1-variant, P2-variant) pair
OUTPUT_DIR        = "sweep_results"
DQN_MODEL_PATH    = "checkpoints/dqn_final.pth"   # fallback DQN path

# ══════════════════════════════════════════════════════════════════════
#  Build variant list  — one entry per (agent_name, params_dict) combo
# ══════════════════════════════════════════════════════════════════════

def build_variants() -> list[tuple[str, dict]]:
    variants = []
    for agent_name, param_grid in PARAM_SWEEP.items():
        if not param_grid:
            variants.append((agent_name, {}))
        else:
            keys   = list(param_grid.keys())
            values = list(param_grid.values())
            for combo in itertools.product(*values):
                variants.append((agent_name, dict(zip(keys, combo))))
    return variants


def variant_label(name: str, params: dict) -> str:
    if not params:
        return name
    tag = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{name}[{tag}]"


# ══════════════════════════════════════════════════════════════════════
#  Agent factory
# ══════════════════════════════════════════════════════════════════════

def make_agent(name: str, player_id: int, params: dict):
    if name == "Random":
        return random_agent.RandomAgent(player_id)
    elif name == "RuleBased":
        return rule_based_agent.RuleBasedAgent(player_id)
    elif name == "MinMax":
        return minmax_agent.MinMaxAgent(player_id, **params)
    elif name == "AlphaBeta":
        return alfabetapruning_agent.AlphaBetaAgent(player_id, **params)
    elif name == "MCTS":
        return mcts_agent.MCTSAgent(player_id=player_id, **params)
    elif name == "DQN":
        model_path = params.get("model_path", DQN_MODEL_PATH)
        agent = DQNAgent(player_id)
        try:
            # Use the agent's own load() — auto-detects legacy vs current architecture
            agent.load(model_path)
            agent.epsilon = 0.0
        except FileNotFoundError:
            print(f"  [WARNING] DQN weights not found at '{model_path}' — playing randomly.")
        return agent
    else:
        raise ValueError(f"Unknown agent: '{name}'")


# ══════════════════════════════════════════════════════════════════════
#  Single game
# ══════════════════════════════════════════════════════════════════════

def play_game(agent1, agent2) -> tuple[str, float]:
    game = connect4.Connect4Game()
    t0   = time.time()

    while True:
        current = agent1 if game.current_player == 1 else agent2
        move    = current.select_move(game)

        if move is None or move == -1:
            outcome = "Draw" if game.check_draw() else f"Error_{game.current_player}"
            return outcome, time.time() - t0

        if not (0 <= move < game.column_count) or not game.is_valid_location(game.board, move):
            return f"Error_{game.current_player}", time.time() - t0

        ok, row = game.make_move(move)
        if not ok:
            return f"Error_{game.current_player}", time.time() - t0

        if game.check_winner(row, move, game.current_player):
            return f"Player_{game.current_player}", time.time() - t0

        if game.check_draw():
            return "Draw", time.time() - t0

        game.switch_player()


# ══════════════════════════════════════════════════════════════════════
#  One matchup
# ══════════════════════════════════════════════════════════════════════

def run_matchup(
    p1_name:   str, p1_params: dict,
    p2_name:   str, p2_params: dict,
    num_games: int,
    idx: int,  total: int,
) -> dict:
    lbl1 = variant_label(p1_name, p1_params)
    lbl2 = variant_label(p2_name, p2_params)

    print(f"\n  [{idx}/{total}]  {lbl1}  vs  {lbl2}  ({num_games} games)")

    agent1 = make_agent(p1_name, 1, p1_params)
    agent2 = make_agent(p2_name, 2, p2_params)

    totals  = {"Player_1": 0, "Player_2": 0, "Draw": 0}
    lines   = []
    t_start = time.time()

    for i in range(1, num_games + 1):
        outcome, dur = play_game(agent1, agent2)
        totals[outcome] = totals.get(outcome, 0) + 1
        lines.append(f"Game {i:02d}: {outcome:<14}  ({dur:.2f}s)")
        print(f"    game {i:02d}/{num_games}  ->  {outcome}  ({dur:.2f}s)")

    total_time = time.time() - t_start
    wr1 = totals["Player_1"] / num_games
    wr2 = totals["Player_2"] / num_games

    safe = lambda s: s.replace("/", "-").replace("\\", "-").replace(":", "=")
    filename = f"{safe(lbl1)}_vs_{safe(lbl2)}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 56 + "\n")
        f.write(f"  P1 : {lbl1}\n")
        f.write(f"  P2 : {lbl2}\n")
        f.write(f"  P1 params : {p1_params}\n")
        f.write(f"  P2 params : {p2_params}\n")
        f.write(f"  Games     : {num_games}\n")
        f.write("=" * 56 + "\n\n")
        f.write("\n".join(lines) + "\n\n")
        f.write("-" * 56 + "\n")
        f.write(f"  P1 wins  : {totals['Player_1']}  ({wr1:.0%})\n")
        f.write(f"  P2 wins  : {totals['Player_2']}  ({wr2:.0%})\n")
        f.write(f"  Draws    : {totals['Draw']}\n")
        f.write(f"  Time     : {total_time:.2f}s\n")
        f.write("-" * 56 + "\n")

    print(f"    -> {totals['Player_1']}W / {totals['Player_2']}W / {totals['Draw']}D  [{total_time:.1f}s]")

    return {
        "p1_agent":   p1_name,
        "p1_params":  str(p1_params),
        "p1_label":   lbl1,
        "p2_agent":   p2_name,
        "p2_params":  str(p2_params),
        "p2_label":   lbl2,
        "games":      num_games,
        "p1_wins":    totals["Player_1"],
        "p2_wins":    totals["Player_2"],
        "draws":      totals["Draw"],
        "p1_winrate": round(wr1, 4),
        "p2_winrate": round(wr2, 4),
        "time_s":     round(total_time, 2),
        "file":       filepath,
    }


# ══════════════════════════════════════════════════════════════════════
#  Summary helpers
# ══════════════════════════════════════════════════════════════════════

def save_summary_csv(rows: list[dict]):
    path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV summary -> {path}")


def print_best_params(rows: list[dict]):
    print(f"\n{'=' * 62}")
    print("  BEST PARAMS PER AGENT  (avg P1 win-rate across all P2 opponents)")
    print(f"{'=' * 62}")

    agents = sorted(set(r["p1_agent"] for r in rows))
    for agent in agents:
        agent_rows = [r for r in rows if r["p1_agent"] == agent]

        groups: dict[str, list[float]] = defaultdict(list)
        for r in agent_rows:
            groups[r["p1_params"]].append(r["p1_winrate"])

        if not groups:
            continue

        best_combo  = max(groups, key=lambda k: sum(groups[k]) / len(groups[k]))
        best_wr     = sum(groups[best_combo])  / len(groups[best_combo])
        worst_combo = min(groups, key=lambda k: sum(groups[k]) / len(groups[k]))
        worst_wr    = sum(groups[worst_combo]) / len(groups[worst_combo])

        print(f"\n  {agent}  ({len(groups)} param combos tested)")
        print(f"    best   {best_combo}")
        print(f"           avg win-rate {best_wr:.1%}  over {len(groups[best_combo])} matchups")
        print(f"    worst  {worst_combo}")
        print(f"           avg win-rate {worst_wr:.1%}  over {len(groups[worst_combo])} matchups")

    print(f"\n{'=' * 62}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    variants = build_variants()
    matchups = list(itertools.product(variants, variants))
    total    = len(matchups)

    agents_seen: dict[str, int] = defaultdict(int)
    for name, _ in variants:
        agents_seen[name] += 1

    print("+" + "=" * 60 + "+")
    print("|  Connect-4 Parameter Sweep — All Variants vs All Variants  |")
    print("+" + "=" * 60 + "+")
    for name, count in agents_seen.items():
        line = f"|  {name:<12}  {count:>3} variant(s)"
        print(line + " " * (62 - len(line)) + "|")
    print("+" + "=" * 60 + "+")
    print(f"|  Total variants  : {len(variants):<41}|")
    print(f"|  Total matchups  : {total:<41}|")
    print(f"|  Games/matchup   : {GAMES_PER_MATCHUP:<41}|")
    print(f"|  Total games     : {total * GAMES_PER_MATCHUP:<41}|")
    print("+" + "=" * 60 + "+")

    global_start = time.time()
    all_results: list[dict] = []

    for idx, ((p1_name, p1_params), (p2_name, p2_params)) in enumerate(matchups, start=1):
        result = run_matchup(
            p1_name, p1_params,
            p2_name, p2_params,
            num_games = GAMES_PER_MATCHUP,
            idx       = idx,
            total     = total,
        )
        all_results.append(result)

    elapsed = time.time() - global_start
    save_summary_csv(all_results)
    print_best_params(all_results)

    print(f"\n  Total time : {elapsed:.1f}s")
    print(f"  All files  : {OUTPUT_DIR}/")