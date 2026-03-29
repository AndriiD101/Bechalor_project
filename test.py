import torch
from game import connect4
from agents import mcts_agent, minmax_agent, rule_based_agent, random_agent, alfabetapruning_agent

from agents.reinforcement_agent import DQNAgent 

def load_dqn_agent(player_id, model_path="dqn_connect4_final.pth"):
    """Допоміжна функція для правильного завантаження DQN-агента"""
    agent = DQNAgent(player_id)
    try:
        # Use the agent's own load() method — it auto-detects legacy vs current architecture
        agent.load(model_path)
        
        # Повністю вимикаємо випадкові дослідження – агент робить лише найкращі ходи
        agent.epsilon = 0.0     
        print(f"[УСПІХ] DQN модель завантажена з '{model_path}'!")
    except FileNotFoundError:
        print(f"[УВАГА] Файл '{model_path}' не знайдено. Агент гратиме випадково (ненавчений).")
    
    return agent

if __name__ == "__main__":
    game = connect4.Connect4Game()
    
    # --- ВИБІР АГЕНТІВ ---
    
    # Гравець 1: Наш навчений DQN
    # agent1 = load_dqn_agent(1, "checkpoints/dqn_final.pth")
    # agent1 = rule_based_agent.RuleBasedAgent(1)
    # agent1 = alfabetapruning_agent.AlphaBetaAgent(1, max_depth=6)
    # agent1 = minmax_agent.MinMaxAgent(1, max_depth=8) 
    agent1 = mcts_agent.MCTSAgent(1, max_iterations=30000)
    
    # Гравець 2: Обирай супротивника (просто розкоментуй потрібного)
    # agent2 = random_agent.RandomAgent(2)
    # agent2 = rule_based_agent.RuleBasedAgent(2)
    agent2 = minmax_agent.MinMaxAgent(2, max_depth=4) 
    # agent2 = alfabetapruning_agent.AlphaBetaAgent(2, max_depth=8)
    # agent2 = mcts_agent.MCTSAgent(2)
    
    print(f"Починаємо гру: Гравець 1 ({type(agent1).__name__}) VS Гравець 2 ({type(agent2).__name__})")
    game.print_board()
    
    while True:
        current_agent = agent1 if game.current_player == 1 else agent2
        
        move = current_agent.select_move(game)
        
        if move == -1:
            game.check_draw()
            print("Draw")
            break
        
        success, row = game.make_move(move)

        if not success:
            print(f"Player {game.current_player} ({type(current_agent).__name__}) зробив неможливий хід у колонку {move}. Автоматична поразка.")
            break

        game.print_board()

        if game.check_winner(row, move, game.current_player):
            print(f"Player {game.current_player} ({type(current_agent).__name__}) won!")
            break

        if game.check_draw():
            print("Game ended with draw.")
            break

        game.switch_player()