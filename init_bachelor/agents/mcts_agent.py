import math
import random
import copy
from agents.agents_interface import AgentInterface


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move              # column that led here

        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.wins: float = 0.0       # accumulated reward from this node's perspective

        # Columns not yet expanded into child nodes (use list for faster iteration)
        self.untried_moves: list[int] = game_state.get_valid_locations()

    # ------------------------------------------------------------------ #
    #  Tree-policy helpers                                                 #
    # ------------------------------------------------------------------ #

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        gs = self.game_state
        # Check if the previous player (the one who just moved) won.
        # current_player has already been switched, so the last mover is the opponent.
        last_player = 3 - gs.current_player          # flips 1↔2
        return (
            gs.winning_move(last_player)
            or gs.check_draw()
        )

    def best_child(self, exploration_weight: float = math.sqrt(2)) -> "MCTSNode":
        log_parent = math.log(self.visits)
        exploration_c_sqrt = exploration_weight * math.sqrt(log_parent)

        best_child = self.children[0]
        best_ucb = (best_child.wins / best_child.visits) + (exploration_c_sqrt / math.sqrt(best_child.visits))

        for child in self.children[1:]:
            exploitation = child.wins / child.visits
            exploration = exploration_c_sqrt / math.sqrt(child.visits)
            ucb = exploitation + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def most_visited_child(self) -> "MCTSNode":
        best_child = self.children[0]
        max_visits = best_child.visits
        for child in self.children[1:]:
            if child.visits > max_visits:
                max_visits = child.visits
                best_child = child
        return best_child

    # ------------------------------------------------------------------ #
    #  Expansion                                                           #
    # ------------------------------------------------------------------ #

    def expand(self) -> "MCTSNode":
        # Use pop to avoid O(n) remove operation
        move_idx = random.randint(0, len(self.untried_moves) - 1)
        move = self.untried_moves[move_idx]
        self.untried_moves[move_idx] = self.untried_moves[-1]
        self.untried_moves.pop()

        new_state = self.game_state.clone()
        new_state.make_move(move)
        new_state.switch_player()

        child = MCTSNode(game_state=new_state, parent=self, move=move)
        self.children.append(child)
        return child

    # ------------------------------------------------------------------ #
    #  Back-propagation                                                    #
    # ------------------------------------------------------------------ #

    def backpropagate(self, result: float):
        self.visits += 1
        self.wins   += result
        if self.parent is not None:
            self.parent.backpropagate(1.0 - result)


# ======================================================================= #
#  MCTS Agent                                                               #
# ======================================================================= #

class MCTSAgent(AgentInterface):
    """
    Monte Carlo Tree Search agent for Connect-4.

    Conforms to AgentInterface: implements `select_move(game) -> int`.

    Algorithm outline
    -----------------
    Repeat `num_simulations` times:
      1. **Selection**  – traverse the tree with UCB1 until a non-terminal,
                          not-fully-expanded node is found.
      2. **Expansion**  – add one new child node for an untried move.
      3. **Simulation** – play out the game randomly from the new node.
      4. **Back-prop**  – propagate the result back up the tree.
    Return the move of the most-visited root child.

    Parameters
    ----------
    player_id       : 1 or 2
    num_simulations : budget per call to select_move (default 1000)
    exploration_c   : UCB1 exploration constant C (default √2 ≈ 1.414)
    """

    def __init__(
        self,
        player_id: int,
        max_iterations: int = 1000,
        exploration_c: float = math.sqrt(2),
    ):
        super().__init__(player_id)
        self.max_iterations = max_iterations
        self.exploration_c   = exploration_c

    # ------------------------------------------------------------------ #
    #  Public API (AgentInterface)                                         #
    # ------------------------------------------------------------------ #

    def select_move(self, game) -> int:
        root = MCTSNode(game_state=game.clone())
        exploration_c = self.exploration_c

        for _ in range(self.max_iterations):
            node = self._select(root, exploration_c)

            if not node.is_terminal():
                node = node.expand()

            result = self._simulate(node)
            node.backpropagate(result)

        best = root.most_visited_child()
        return best.move

    # ------------------------------------------------------------------ #
    #  MCTS phases                                                         #
    # ------------------------------------------------------------------ #

    def _select(self, node: MCTSNode, exploration_c: float = None) -> MCTSNode:
        if exploration_c is None:
            exploration_c = self.exploration_c
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node          # stop here; let expand() handle it
            node = node.best_child(exploration_c)
        return node

    def _simulate(self, node: MCTSNode) -> float:
        sim_state = node.game_state.clone()
        player_id = self.player_id

        while True:
            # Check whether the last move caused a win
            last_player = 3 - sim_state.current_player   # player who just moved
            if sim_state.winning_move(last_player):
                # Return result from the perspective of `player_id`
                return 1.0 if last_player == player_id else 0.0

            if sim_state.check_draw():
                return 0.5

            valid = sim_state.get_valid_locations()
            # Use direct indexing instead of random.choice for faster access
            move = valid[random.randint(0, len(valid) - 1)]
            sim_state.make_move(move)
            sim_state.switch_player()