"""
Snake Eater - Manual Play Mode
- Smooth movement with grid-locked turns and filled corners
- Snake starts length 3 in the center
- Press R or SPACE to restart (after game over or anytime)
- Press ESC to exit
- Direction inputs are queued (no missed turns)
- Game waits for first input before starting
"""

import pygame
import sys
import random
from collections import deque

try:
    from algorithms.hamilton_cycle import HamiltonPathPlanner
except ImportError:
    HamiltonPathPlanner = None


def play_game(grid_width=10, grid_height=10, speed_cells=6, cell_size=None, fps=60):
    """
    Main manual play game function

    Args:
        grid_width: Width of the grid
        grid_height: Height of the grid
        speed_cells: Movement speed in cells per second
        cell_size: Size of each cell in pixels (None = auto-calculate to fit screen)
        fps: Frames per second
    """
    # ---------------- Settings ----------------
    GRID_W, GRID_H = grid_width, grid_height
    SPEED_CELLS = speed_cells
    FPS = fps

    # Auto-calculate cell size to fit screen if not specified
    if cell_size is None:
        # Initialize pygame to get display info
        pygame.init()
        display_info = pygame.display.Info()
        screen_width = display_info.current_w
        screen_height = display_info.current_h

        # Leave some margin (90% of screen size)
        max_width = int(screen_width * 0.9)
        max_height = int(screen_height * 0.9)

        # Calculate cell size that fits both dimensions
        cell_by_width = max_width // GRID_W
        cell_by_height = max_height // GRID_H
        cell_size = min(cell_by_width, cell_by_height, 60)  # Cap at 60px for small grids

        print(f"Auto-calculated cell size: {cell_size}px (Window: {cell_size * GRID_W}x{cell_size * GRID_H})")

    CELL = cell_size
    frame_size_x = GRID_W * CELL
    frame_size_y = GRID_H * CELL

    SNAKE_THICKNESS = int(CELL * 0.98)

    # ---------------- Init ----------------
    check_errors = pygame.init()
    if check_errors[1] > 0:
        print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
        sys.exit(-1)

    pygame.display.set_caption('Snake Eater')
    # Make window resizable
    game_window = pygame.display.set_mode((frame_size_x, frame_size_y), pygame.RESIZABLE)
    is_fullscreen = False

    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red   = pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    grid_col = pygame.Color(40, 40, 40)

    fps_controller = pygame.time.Clock()

    DIRS = {
        'UP':    (0, -1),
        'DOWN':  (0,  1),
        'LEFT':  (-1, 0),
        'RIGHT': (1,  0),
    }

    # Initialize Hamilton path planner
    show_hamilton_path = False
    hamilton_planner = None
    if HamiltonPathPlanner is not None:
        try:
            hamilton_planner = HamiltonPathPlanner(GRID_W, GRID_H)
            print("Hamilton Cycle visualization available - Press H to toggle")
        except Exception as e:
            print(f"Could not initialize Hamilton path: {e}")
    else:
        print("Hamilton Cycle not available (algorithms/hamilton_cycle.py not found)")

    def opposite_dir(a, b):
        return (a[0] == -b[0] and a[1] == -b[1])

    def cell_center(cell_xy):
        return (cell_xy[0] * CELL + CELL / 2, cell_xy[1] * CELL + CELL / 2)

    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def draw_grid(surface):
        for x in range(GRID_W + 1):
            pygame.draw.line(surface, grid_col, (x * CELL, 0), (x * CELL, frame_size_y))
        for y in range(GRID_H + 1):
            pygame.draw.line(surface, grid_col, (0, y * CELL), (frame_size_x, y * CELL))

    def random_food(occupied):
        empties = [(x, y) for x in range(GRID_W) for y in range(GRID_H) if [x, y] not in occupied]
        return list(random.choice(empties)) if empties else None

    def draw_snake_with_filled_corners(surface, points, thickness, color):
        if not points:
            return
        half = thickness / 2

        # draw axis-aligned segment rectangles
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) >= abs(dy):  # horizontal
                left = min(x1, x2)
                width = abs(dx)
                rect = pygame.Rect(int(left), int(y1 - half), int(width), int(thickness))
            else:                   # vertical
                top = min(y1, y2)
                height = abs(dy)
                rect = pygame.Rect(int(x1 - half), int(top), int(thickness), int(height))
            pygame.draw.rect(surface, color, rect)

        # fill joints with boxes (perfect elbows)
        s = int(thickness)
        for (cx, cy) in points:
            rect = pygame.Rect(int(cx - half), int(cy - half), s, s)
            pygame.draw.rect(surface, color, rect)

    def reset_game():
        center_x = GRID_W // 2
        center_y = GRID_H // 2
        snake_body = [[center_x, center_y], [center_x-1, center_y], [center_x-2, center_y]]
        prev_body = [seg[:] for seg in snake_body]
        direction = 'RIGHT'
        food_pos = random_food(snake_body)
        score = 0
        game_over = False
        waiting_to_start = True
        accum = 0.0
        dir_queue = deque()  # <- input queue
        return snake_body, prev_body, direction, food_pos, score, game_over, waiting_to_start, accum, dir_queue

    # fixed-step timing
    step_time = 1.0 / SPEED_CELLS

    # font
    score_font = pygame.font.SysFont('consolas', 20)
    over_font = pygame.font.SysFont('times new roman', 48)
    start_font = pygame.font.SysFont('times new roman', 36)

    snake_body, prev_body, direction, food_pos, score, game_over, waiting_to_start, accum, dir_queue = reset_game()

    KEY_TO_DIR = {
        pygame.K_UP: 'UP', pygame.K_w: 'UP',
        pygame.K_DOWN: 'DOWN', pygame.K_s: 'DOWN',
        pygame.K_LEFT: 'LEFT', pygame.K_a: 'LEFT',
        pygame.K_RIGHT: 'RIGHT', pygame.K_d: 'RIGHT',
    }

    MAX_QUEUE = 3  # keep it small so it feels responsive but not "buffered forever"

    def queue_dir(d):
        """Queue direction if it's not a duplicate of the last queued item."""
        if len(dir_queue) >= MAX_QUEUE:
            return
        if dir_queue and dir_queue[-1] == d:
            return
        dir_queue.append(d)

    def consume_next_valid_dir(current_direction):
        """
        Pop directions until one is valid (not opposite of current_direction).
        Return updated_direction (or current if none valid).
        """
        cur_vec = DIRS[current_direction]
        while dir_queue:
            cand = dir_queue.popleft()
            cand_vec = DIRS[cand]
            if not opposite_dir(cand_vec, cur_vec):
                return cand
        return current_direction

    def recalculate_display(new_width, new_height):
        """Recalculate cell size and thickness based on new window size"""
        nonlocal CELL, SNAKE_THICKNESS, frame_size_x, frame_size_y

        # Calculate new cell size that fits the window
        cell_by_width = new_width // GRID_W
        cell_by_height = new_height // GRID_H
        CELL = min(cell_by_width, cell_by_height)

        # Recalculate dimensions
        frame_size_x = GRID_W * CELL
        frame_size_y = GRID_H * CELL
        SNAKE_THICKNESS = int(CELL * 0.98)

        return CELL

    # ---------------- Main loop ----------------
    while True:
        dt = fps_controller.tick(FPS) / 1000.0
        if not waiting_to_start:
            accum += dt

        # ---- Input ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle window resize
            if event.type == pygame.VIDEORESIZE:
                if not is_fullscreen:
                    recalculate_display(event.w, event.h)
                    game_window = pygame.display.set_mode((frame_size_x, frame_size_y), pygame.RESIZABLE)

            if event.type == pygame.KEYDOWN:
                # R or SPACE restarts anytime
                if event.key == pygame.K_r or event.key == pygame.K_SPACE:
                    snake_body, prev_body, direction, food_pos, score, game_over, waiting_to_start, accum, dir_queue = reset_game()
                    continue

                # ESC exits
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

                # H toggles Hamilton path visibility
                if event.key == pygame.K_h:
                    if hamilton_planner is not None:
                        show_hamilton_path = not show_hamilton_path
                        status = "ON" if show_hamilton_path else "OFF"
                        print(f"Hamilton path overlay: {status}")

                # F11 toggles fullscreen
                if event.key == pygame.K_F11:
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        # Get screen resolution for fullscreen
                        display_info = pygame.display.Info()
                        screen_width = display_info.current_w
                        screen_height = display_info.current_h
                        recalculate_display(screen_width, screen_height)
                        game_window = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
                        print("Fullscreen: ON (Press F11 to exit)")
                    else:
                        # Return to windowed mode with calculated size
                        game_window = pygame.display.set_mode((frame_size_x, frame_size_y), pygame.RESIZABLE)
                        print("Fullscreen: OFF")

                # Movement keys
                if event.key in KEY_TO_DIR:
                    if waiting_to_start:
                        # Start the game with the first input
                        waiting_to_start = False
                        direction = KEY_TO_DIR[event.key]
                    elif not game_over:
                        queue_dir(KEY_TO_DIR[event.key])

        # ---- Update (only if not game over and not waiting) ----
        if not game_over and not waiting_to_start:
            while accum >= step_time:
                accum -= step_time
                prev_body = [seg[:] for seg in snake_body]

                # Apply ONE queued turn per step (as soon as possible)
                direction = consume_next_valid_dir(direction)

                dx, dy = DIRS[direction]
                head_x, head_y = snake_body[0]
                new_head = [head_x + dx, head_y + dy]

                # collisions
                if new_head[0] < 0 or new_head[0] >= GRID_W or new_head[1] < 0 or new_head[1] >= GRID_H:
                    game_over = True
                    break
                if new_head in snake_body:
                    game_over = True
                    break

                snake_body.insert(0, new_head)

                # food
                if food_pos and new_head == food_pos:
                    score += 1
                    food_pos = random_food(snake_body)
                else:
                    snake_body.pop()

        # ---- Render ----
        game_window.fill(black)

        # Draw Hamilton path as subtle white line in background (if enabled)
        if show_hamilton_path and hamilton_planner is not None:
            for y in range(GRID_H):
                for x in range(GRID_W):
                    current_idx = hamilton_planner.path_map[y, x]
                    next_idx = (current_idx + 1) % (GRID_W * GRID_H)

                    # Find next cell in path
                    for ny in range(GRID_H):
                        for nx in range(GRID_W):
                            if hamilton_planner.path_map[ny, nx] == next_idx:
                                # Draw thin white line from current to next
                                start = (x * CELL + CELL // 2, y * CELL + CELL // 2)
                                end = (nx * CELL + CELL // 2, ny * CELL + CELL // 2)
                                pygame.draw.line(game_window, white, start, end, 2)
                                break
                        else:
                            continue
                        break

        alpha = 0.0 if (game_over or waiting_to_start) else max(0.0, min(accum / step_time, 1.0))

        if len(prev_body) < len(snake_body):
            prev_body = prev_body + [prev_body[-1][:]] * (len(snake_body) - len(prev_body))

        interp_centers = []
        for prev_seg, cur_seg in zip(prev_body, snake_body):
            pa = cell_center(prev_seg)
            pb = cell_center(cur_seg)
            interp_centers.append(lerp(pa, pb, alpha))

        # insert corner points so elbows are filled boxes
        tube_points = [interp_centers[0]]
        for i in range(1, len(interp_centers)):
            corner = cell_center(prev_body[i - 1])
            if (abs(tube_points[-1][0] - corner[0]) > 0.01) or (abs(tube_points[-1][1] - corner[1]) > 0.01):
                tube_points.append(corner)
            tube_points.append(interp_centers[i])

        draw_snake_with_filled_corners(game_window, tube_points, SNAKE_THICKNESS, green)

        if food_pos:
            food_size = int(CELL * 0.98)
            offset = (CELL - food_size) // 2
            pygame.draw.rect(
                game_window, red,
                pygame.Rect(food_pos[0] * CELL + offset, food_pos[1] * CELL + offset, food_size, food_size)
            )

        game_window.blit(score_font.render(f'Score : {score}', True, red), (10, 10))

        if waiting_to_start:
            msg = start_font.render("Press Arrow Key to Start", True, white)
            rect = msg.get_rect(center=(frame_size_x // 2, frame_size_y // 2))
            game_window.blit(msg, rect)
        elif game_over:
            msg = over_font.render("GAME OVER", True, red)
            rect = msg.get_rect(center=(frame_size_x // 2, frame_size_y // 2))
            game_window.blit(msg, rect)

        pygame.display.update()


if __name__ == "__main__":
    # Run with default settings when executed directly
    play_game()
