def check_diagonal(board, row, col, piece):
    # /
    count = 0
    r, c = row, col
    while r > 0 and c > 0:
        r -= 1
        c -= 1
    while r < ROW_COUNT and c < COLUMN_COUNT:
        if board[r][c] == piece:
            count += 1
            if count == 4:
                return True
        else:
            count = 0
        r += 1
        c += 1

    # \
    count = 0
    r, c = row, col
    while r < ROW_COUNT - 1 and c > 0:
        r += 1
        c -= 1
    while r >= 0 and c < COLUMN_COUNT:
        if board[r][c] == piece:
            count += 1
            if count == 4:
                return True
        else:
            count = 0
        r -= 1
        c += 1

    return False

for i in range(10, 0, -1):
    print(i)