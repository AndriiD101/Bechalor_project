import pygame
import math
import sys
import numpy as np
import copy

class Connect4Game:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.board = np.zeros((self.row_count, self.column_count))
        self.current_player = 1
    
    def is_valid_location(self, board, col):
        if board[self.row_count-1][col] == 0:
            return True
        return False
    
    def make_move(self, col) -> tuple[bool, int | None]:
        row = self.get_next_open_row(col)
        if row is not None:
            self.board[row][col] = self.current_player
            return True, row
        return False, None
    
    def get_next_open_row(self, col):
        for row in range(self.row_count):
            if self.board[row][col] == 0:
                return row
        return None
    
    def check_horizontal(self, row, player):
        count = 0
        for col in range(self.column_count):
            if self.board[row][col] == player:
                count += 1
                if count == 4:
                    return player, True
            else:
                count = 0
        return player, False

    def check_vertical(self, col, player):
        count = 0
        for row in range(self.row_count):
            if self.board[row][col] == player:
                count += 1
                if count == 4:
                    return player, True
            else:
                count = 0
        return player, False
    
    def check_diagonal(self, row, col, player):
        #diagonal check
        # Bottom-left to top-right (/)
        count = 0
        start_row, start_col = row, col
        while start_row > 0 and start_col > 0:
            start_row -= 1
            start_col -= 1
        while start_row < self.row_count and start_col < self.column_count:
            if self.board[start_row][start_col] == player:
                count += 1
                if count == 4:
                    return player, True
            else:
                count = 0
            start_row += 1
            start_col += 1

        # Top-left to bottom-right (\)
        count = 0
        start_row, start_col = row, col
        while start_row < self.row_count - 1 and start_col > 0:
            start_row += 1
            start_col -= 1
        while start_row >= 0 and start_col < self.column_count:
            if self.board[start_row][start_col] == player:
                count += 1
                if count == 4:
                    return player, True
            else:
                count = 0
            start_row -= 1
            start_col += 1

        return player, False
            
    def check_winner(self, row, col, player):
        _, h = self.check_horizontal(row, player)
        _, v = self.check_vertical(col, player)
        _, d = self.check_diagonal(row, col, player)
        return h or v or d

    def check_draw(self):
        return np.all(self.board != 0)
    
    def switch_player(self):
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1
     
    def clone(self):
        new_game = Connect4Game()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        return new_game
    
    def print_board(self):
        print(self.board)
     
if __name__ == "__main__":
    pass