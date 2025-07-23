from game import connect4
from agents import minmax_agent, rule_based_agent, random_agent
from time import sleep

if __name__ == "__main__":
    # Create a game
    game = connect4.Connect4Game()
    random_agent1 = minmax_agent.MinMaxAgent(1)
    random_agent2 = random_agent.RandomAgent(2)
    
    game.print_board()
    
    while True:
        current_agent = random_agent1 if game.current_player == 1 else random_agent2
        move =  current_agent.select_move(game)
        
        if move == -1:
            game.check_draw()
            print("Draw")
            break
        
        success = game.make_move(move)

        if not success:
            print(f"Гравець {game.current_player} ({current_agent}) зробив некоректний хід. Програв автоматично.")
            break

        game.print_board()

        # Перевірка переможця
        row = game.get_next_open_row(move)
        row = row - 1 if row is not None and row > 0 else 0
        if game.check_winner(row, move, game.current_player):
            print(f"Гравець {game.current_player} ({current_agent}) переміг!")
            break

        # Перевірка нічиєї
        if game.check_draw():
            print("Гра завершилася нічиєю.")
            break

        game.switch_player()

# if __name__ == "__main__":
#     game = connect4.Connect4Game()
#     agent = minmax_agent.MinMaxAgent(1)
    
#     agent.select_move(game)