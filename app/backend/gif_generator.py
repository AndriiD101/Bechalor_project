import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generate_connect4_gif(move_history, filepath, p1_name="Player 1", p2_name="Player 2"):
    ROWS = 6
    COLS = 7
    CELL_SIZE = 60
    PADDING = 10
    HEADER_HEIGHT = 45 
    
    BOARD_WIDTH = COLS * CELL_SIZE + PADDING * 2
    BOARD_HEIGHT = ROWS * CELL_SIZE + PADDING * 2
    TOTAL_HEIGHT = BOARD_HEIGHT + HEADER_HEIGHT 

    # Colors
    BG_COLOR = (18, 18, 28)
    BORDER_COLOR = (42, 42, 69)
    P1_COLOR = (247, 201, 72)
    P2_COLOR = (255, 77, 109)
    EMPTY_COLOR = (30, 30, 46)
    TEXT_COLOR = (200, 200, 220)
    WIN_BLINK_COLOR = (46, 204, 113) # ➔ A nice neon/cyberpunk green!

    board = np.zeros((ROWS, COLS), dtype=int)
    frames = []
    
    matchup_text = f"{p1_name.upper()} vs {p2_name.upper()}"

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20) 
        except IOError:
            font = ImageFont.load_default()

    # ➔ Helper function to find the winning 4 pieces
    def get_winning_pieces(current_board):
        for r in range(ROWS):
            for c in range(COLS):
                piece = current_board[r][c]
                if piece == 0:
                    continue
                # Check right, up, up-right, down-right
                directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
                for dr, dc in directions:
                    line = [(r, c)]
                    for i in range(1, 4):
                        nr, nc = r + dr * i, c + dc * i
                        if 0 <= nr < ROWS and 0 <= nc < COLS and current_board[nr][nc] == piece:
                            line.append((nr, nc))
                        else:
                            break
                    if len(line) == 4:
                        return line # Return the coordinates of the 4 winning pieces
        return []

    # ➔ Updated draw_board to accept highlighted coordinates
    def draw_board(current_board, highlight_coords=None, highlight_color=None):
        if highlight_coords is None:
            highlight_coords = []

        img = Image.new('RGB', (BOARD_WIDTH, TOTAL_HEIGHT), color=BG_COLOR)
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, BOARD_WIDTH-1, TOTAL_HEIGHT-1], outline=BORDER_COLOR, width=2)

        try:
            bbox = draw.textbbox((0, 0), matchup_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(matchup_text, font=font)

        text_x = (BOARD_WIDTH - text_w) // 2
        text_y = (HEADER_HEIGHT - text_h) // 2
        draw.text((text_x, text_y), matchup_text, fill=TEXT_COLOR, font=font)

        for r in range(ROWS):
            for c in range(COLS):
                visual_r = ROWS - 1 - r

                x0 = PADDING + c * CELL_SIZE + 5
                y0 = HEADER_HEIGHT + PADDING + visual_r * CELL_SIZE + 5 
                x1 = x0 + CELL_SIZE - 10
                y1 = y0 + CELL_SIZE - 10

                val = current_board[r][c]
                
                # ➔ Apply the blink color if this piece is in the winning line
                if (r, c) in highlight_coords:
                    fill = highlight_color
                elif val == 1:
                    fill = P1_COLOR
                elif val == 2:
                    fill = P2_COLOR
                else:
                    fill = EMPTY_COLOR

                draw.ellipse([x0, y0, x1, y1], fill=fill, outline=BORDER_COLOR, width=2)
                
        return img

    # --- Generate the frames ---
    
    # Frame 1: The empty board
    frames.append(draw_board(board))

    # Replay actual moves
    for move in move_history:
        p = move["player"]
        r = move["row"]
        c = move["col"]
        board[r][c] = p
        frames.append(draw_board(board))

    # Calculate base gameplay timing
    target_gameplay_ms = 5000
    total_moves = max(1, len(frames))
    base_frame_duration = max(100, int(target_gameplay_ms / total_moves))

    # Set durations for the gameplay phase
    frame_durations = [base_frame_duration] * len(frames)

    # ➔ Add the Blinking Frames at the end
    winning_pieces = get_winning_pieces(board)
    if winning_pieces:
        # Blink 3 times
        for _ in range(3):
            # Frame 1 of blink: Green
            frames.append(draw_board(board, highlight_coords=winning_pieces, highlight_color=WIN_BLINK_COLOR))
            frame_durations.append(300) # Stay green for 300ms
            
            # Frame 2 of blink: Normal color
            frames.append(draw_board(board))
            frame_durations.append(300) # Stay normal for 300ms

    # Final hold frame (so you can stare at the victory before the GIF loops)
    frames.append(frames[-1])
    frame_durations.append(2000) # Hold for 2 seconds

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    frames[0].save(
        filepath,
        save_all=True,
        append_images=frames[1:],
        duration=frame_durations, 
        loop=0
    )
    return filepath