"""
Snake Game Environment for Reinforcement Learning
Supports both headless (fast training) and visual (watch AI play) modes
Based on the smooth movement from snake.py
"""

import pygame
import sys
import random
import numpy as np
try:
    from algorithms.hamilton_cycle import HamiltonPathPlanner
except ImportError:
    HamiltonPathPlanner = None


def cell_center(cell_xy, cell_size):
    """Get pixel center of a grid cell"""
    return (cell_xy[0] * cell_size + cell_size / 2, cell_xy[1] * cell_size + cell_size / 2)


def lerp(a, b, t):
    """Linear interpolation between two points"""
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def draw_snake_with_filled_corners(surface, points, thickness, color):
    """Draw snake with smooth corners (from original snake.py)"""
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


class SnakeEnv:
    def __init__(self, render=False, grid_width=10, grid_height=None, cell_size=None, fps=60, speed_cells=6, use_hamilton=True):
        """
        Initialize Snake Environment

        Args:
            render: If True, show pygame window (for watching AI play)
            grid_width: Width of the grid (x dimension)
            grid_height: Height of the grid (y dimension). If None, uses grid_width (square grid)
            cell_size: Size of each cell in pixels (None = auto-calculate to fit screen, only matters if render=True)
            fps: Frames per second (only matters if render=True)
            speed_cells: Movement speed in cells per second
            use_hamilton: If True, enable Hamilton cycle guidance
        """
        self.render_mode = render
        if grid_height is None:
            grid_height = grid_width
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_size = grid_width  # Keep for backward compatibility

        # Auto-calculate cell size to fit screen if rendering and not specified
        if render and cell_size is None:
            pygame.init()
            display_info = pygame.display.Info()
            screen_width = display_info.current_w
            screen_height = display_info.current_h

            # Leave some margin (90% of screen size)
            max_width = int(screen_width * 0.9)
            max_height = int(screen_height * 0.9)

            # Calculate cell size that fits both dimensions
            cell_by_width = max_width // grid_width
            cell_by_height = max_height // grid_height
            cell_size = min(cell_by_width, cell_by_height, 60)  # Cap at 60px for small grids

            print(f"Auto-calculated cell size: {cell_size}px (Window: {cell_size * grid_width}x{cell_size * grid_height})")
        elif cell_size is None:
            cell_size = 60  # Default for non-rendering mode

        self.cell_size = cell_size
        self.fps = fps
        self.speed_cells = speed_cells
        self.use_hamilton = use_hamilton
        self.show_hamilton_path = False  # Can be set externally to show path

        # Initialize Hamilton path planner if available
        if use_hamilton and HamiltonPathPlanner is not None:
            self.hamilton_planner = HamiltonPathPlanner(grid_width, grid_height)
            print("Hamilton Cycle guidance enabled")
        else:
            self.hamilton_planner = None

        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = 4

        # State space (ENHANCED with Hamilton):
        # - Immediate danger (3): straight, right, left - 1 cell ahead
        # - Lookahead danger (3): straight, right, left - 2 cells ahead
        # - Current direction (4): one-hot encoding
        # - Food direction (4): relative to head
        # - Full grid (100): complete spatial awareness (tail marked differently)
        # - Snake length (1): normalized
        # - Accessible space (1): normalized reachable cells
        # - Can reach tail (1): binary flag
        # - Path length to food (1): normalized BFS distance
        # - Space after eating (1): normalized accessible space if food is eaten
        # - Hamilton path direction (4): one-hot encoding of optimal direction
        # - Should follow Hamilton (1): binary flag indicating if snake should follow path
        # Total: 3 + 3 + 4 + 4 + (width*height) + 1 + 1 + 1 + 1 + 1 + 4 + 1
        self.state_size = 3 + 3 + 4 + 4 + grid_width * grid_height + 1 + 1 + 1 + 1 + 1 + 4 + 1

        self.dirs = {
            0: (0, -1),   # UP
            1: (0, 1),    # DOWN
            2: (-1, 0),   # LEFT
            3: (1, 0),    # RIGHT
        }

        # Initialize pygame only if rendering
        if self.render_mode:
            pygame.init()
            self.frame_width = grid_width * cell_size
            self.frame_height = grid_height * cell_size
            # Make window resizable
            self.game_window = pygame.display.set_mode((self.frame_width, self.frame_height), pygame.RESIZABLE)
            pygame.display.set_caption('Snake AI')
            self.fps_controller = pygame.time.Clock()
            self.is_fullscreen = False

            # Colors
            self.black = pygame.Color(0, 0, 0)
            self.white = pygame.Color(255, 255, 255)
            self.red = pygame.Color(255, 0, 0)
            self.green = pygame.Color(0, 255, 0)
            self.grid_col = pygame.Color(40, 40, 40)

            # Font
            self.score_font = pygame.font.SysFont('consolas', 20)

            self.snake_thickness = int(cell_size * 0.98)

            # Smooth movement timing
            self.step_time = 1.0 / speed_cells
            self.accum = 0.0

        self.reset()

    def _recalculate_display(self, new_width, new_height):
        """Recalculate cell size based on new window size"""
        if not self.render_mode:
            return

        # Calculate new cell size that fits the window
        cell_by_width = new_width // self.grid_width
        cell_by_height = new_height // self.grid_height
        self.cell_size = min(cell_by_width, cell_by_height)

        # Recalculate dimensions
        self.frame_width = self.grid_width * self.cell_size
        self.frame_height = self.grid_height * self.cell_size
        self.snake_thickness = int(self.cell_size * 0.98)

    def reset(self):
        """Reset the game to initial state"""
        # Snake starts at center, length 3, moving right
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.snake_body = [[center_x, center_y], [center_x-1, center_y], [center_x-2, center_y]]
        self.prev_body = [seg[:] for seg in self.snake_body]  # For smooth interpolation
        self.direction = 3  # RIGHT
        self.score = 0
        self.steps = 0
        self.food_pos = self._spawn_food()
        self.game_over = False

        if self.render_mode:
            self.accum = 0.0

        return self._get_state()

    def _spawn_food(self):
        """Spawn food at random empty position"""
        empties = [
            [x, y] for x in range(self.grid_width)
            for y in range(self.grid_height)
            if [x, y] not in self.snake_body
        ]
        return random.choice(empties) if empties else None

    def _is_collision(self, pos):
        """Check if position collides with wall or snake body"""
        # Wall collision
        if pos[0] < 0 or pos[0] >= self.grid_width or pos[1] < 0 or pos[1] >= self.grid_height:
            return True
        # Self collision
        if pos in self.snake_body:
            return True
        return False

    def _is_danger(self, pos):
        """
        Check if position is dangerous (for state representation)
        Excludes the tail since it will move away when snake moves forward (unless food is eaten)
        This gives more accurate danger signals for learning.
        """
        # Wall collision is always dangerous
        if pos[0] < 0 or pos[0] >= self.grid_width or pos[1] < 0 or pos[1] >= self.grid_height:
            return True
        # Self collision - but exclude tail since it will likely move
        # (Only dangerous if we eat food and tail doesn't move, but that's rare)
        if pos in self.snake_body[:-1]:  # Exclude tail (last element)
            return True
        return False

    def _is_danger_lookahead(self, pos):
        """
        Check if position is dangerous 2 steps ahead
        Excludes tail AND second-to-last segment (both will likely move in 2 steps)
        """
        # Wall collision is always dangerous
        if pos[0] < 0 or pos[0] >= self.grid_width or pos[1] < 0 or pos[1] >= self.grid_height:
            return True
        # Exclude last 2 segments (will move away in 2 steps, unless food is eaten twice)
        if len(self.snake_body) > 2 and pos in self.snake_body[:-2]:
            return True
        return False

    def _check_collision_type(self, pos):
        """
        Check collision type for different penalties
        Returns: 'wall', 'self', or None
        """
        # Wall collision
        if pos[0] < 0 or pos[0] >= self.grid_width or pos[1] < 0 or pos[1] >= self.grid_height:
            return 'wall'
        # Self collision
        if pos in self.snake_body:
            return 'self'
        return None

    def _count_safe_moves(self):
        """
        Count how many directions the snake can safely move from current position.
        Used to detect if snake is trapping itself.
        """
        head = self.snake_body[0]
        safe_moves = 0

        # Check all 4 directions
        for direction in range(4):
            dx, dy = self.dirs[direction]
            next_pos = [head[0] + dx, head[1] + dy]

            # Check if this move would be safe
            if not self._is_collision(next_pos):
                safe_moves += 1

        return safe_moves

    def _count_accessible_space(self, start_pos=None):
        """
        Count accessible empty cells from a given position using BFS.
        This is crucial for avoiding dead ends.

        Args:
            start_pos: Starting position (default: snake head)

        Returns:
            Number of reachable empty cells
        """
        if start_pos is None:
            start_pos = self.snake_body[0]

        # BFS to find all reachable empty cells
        visited = set()
        queue = [start_pos]
        visited.add(tuple(start_pos))

        while queue:
            current = queue.pop(0)

            # Check all 4 directions
            for direction in range(4):
                dx, dy = self.dirs[direction]
                next_pos = [current[0] + dx, current[1] + dy]
                next_tuple = tuple(next_pos)

                # Skip if already visited
                if next_tuple in visited:
                    continue

                # Skip if collision (wall or body, excluding tail which will move)
                if self._is_collision(next_pos):
                    continue

                # Add to queue and visited
                visited.add(next_tuple)
                queue.append(next_pos)

        return len(visited)

    def _is_trapped(self):
        """
        Check if snake has moved into an area with no escape (dead end).
        Uses flood fill to check if there's enough reachable space.
        Returns True if trapped (not enough space to move).
        """
        head = self.snake_body[0]
        snake_length = len(self.snake_body)

        # BFS to find all reachable empty cells
        visited = set()
        queue = [head]
        visited.add(tuple(head))

        while queue:
            current = queue.pop(0)

            # If we've found enough space, we're not trapped
            if len(visited) >= snake_length:
                return False

            # Check all 4 directions
            for direction in range(4):
                dx, dy = self.dirs[direction]
                next_pos = [current[0] + dx, current[1] + dy]
                next_tuple = tuple(next_pos)

                # Skip if already visited
                if next_tuple in visited:
                    continue

                # Skip if collision
                if self._is_collision(next_pos):
                    continue

                # Add to queue and visited
                visited.add(next_tuple)
                queue.append(next_pos)

        # If reachable space is less than snake length, we're trapped
        return len(visited) < snake_length

    def _bfs_path_length(self, start, goal, exclude_tail=True):
        """
        Calculate shortest path length between two points using BFS.
        Returns path length, or -1 if no path exists.

        Args:
            start: Starting position [x, y]
            goal: Goal position [x, y]
            exclude_tail: If True, treat tail as passable (it will move)

        Returns:
            Path length (number of steps), or -1 if unreachable
        """
        if start == goal:
            return 0

        visited = set()
        queue = [(start, 0)]  # (position, distance)
        visited.add(tuple(start))

        # Create a set of obstacles
        obstacles = set(tuple(seg) for seg in self.snake_body)
        if exclude_tail and len(self.snake_body) > 0:
            obstacles.discard(tuple(self.snake_body[-1]))

        while queue:
            current, dist = queue.pop(0)

            # Check all 4 directions
            for direction in range(4):
                dx, dy = self.dirs[direction]
                next_pos = [current[0] + dx, current[1] + dy]
                next_tuple = tuple(next_pos)

                # Check if we reached the goal
                if next_pos == goal:
                    return dist + 1

                # Skip if already visited
                if next_tuple in visited:
                    continue

                # Skip if out of bounds
                if next_pos[0] < 0 or next_pos[0] >= self.grid_width or \
                   next_pos[1] < 0 or next_pos[1] >= self.grid_height:
                    continue

                # Skip if obstacle
                if next_tuple in obstacles:
                    continue

                # Add to queue and visited
                visited.add(next_tuple)
                queue.append((next_pos, dist + 1))

        return -1  # No path found

    def _can_reach_tail(self):
        """
        Check if there's a path from head to tail.
        This is important - if we can't reach our tail, we're in trouble!

        Returns:
            True if tail is reachable, False otherwise
        """
        if len(self.snake_body) < 2:
            return True

        head = self.snake_body[0]
        tail = self.snake_body[-1]

        path_length = self._bfs_path_length(head, tail, exclude_tail=True)
        return path_length != -1

    def _get_safe_actions(self):
        """
        Get list of actions that don't immediately result in collision.
        Used for emergency fallback.

        Returns:
            List of safe action indices
        """
        head = self.snake_body[0]
        safe_actions = []

        for action in range(4):
            dx, dy = self.dirs[action]
            next_pos = [head[0] + dx, head[1] + dy]

            # Check if this action would collide
            if not self._is_collision(next_pos):
                # Also check if opposite to current direction (invalid)
                current_vec = self.dirs[self.direction]
                new_vec = self.dirs[action]
                if not (current_vec[0] == -new_vec[0] and current_vec[1] == -new_vec[1]):
                    safe_actions.append(action)

        return safe_actions

    def _get_action_toward_tail(self):
        """
        Get action that moves toward tail.
        Tail-following is a safe strategy when no food path exists.

        Returns:
            Action index, or None if tail is unreachable
        """
        if len(self.snake_body) < 2:
            return None

        head = self.snake_body[0]
        tail = self.snake_body[-1]

        # Use BFS to find path to tail
        # (This is a simplified version - just move in general direction)
        dx = tail[0] - head[0]
        dy = tail[1] - head[1]

        # Prioritize larger distance
        if abs(dx) > abs(dy):
            # Move horizontally
            if dx > 0:
                action = 3  # RIGHT
            else:
                action = 2  # LEFT
        else:
            # Move vertically
            if dy > 0:
                action = 1  # DOWN
            else:
                action = 0  # UP

        # Check if this action is safe
        test_pos = [head[0] + self.dirs[action][0], head[1] + self.dirs[action][1]]
        if not self._is_collision(test_pos):
            return action

        # If not safe, try perpendicular directions
        safe_actions = self._get_safe_actions()
        return safe_actions[0] if safe_actions else None

    def _simulate_eating_food(self):
        """
        Simulate eating food and count accessible space after.
        This helps the AI avoid eating food that would trap it.

        Returns:
            Accessible space after hypothetically eating food
        """
        # If no food available, return current accessible space
        if self.food_pos is None:
            return self._count_accessible_space(self.snake_body[0])
        
        # Temporarily add food position as new head
        temp_head = self.food_pos[:]
        temp_body = [temp_head] + self.snake_body[:]  # Don't remove tail (snake grows)

        # Temporarily update snake body
        old_body = self.snake_body
        self.snake_body = temp_body

        # Count accessible space
        space = self._count_accessible_space(temp_head)

        # Restore original body
        self.snake_body = old_body

        return space

    def _get_state(self):
        """
        Get current state representation for the neural network (ENHANCED)
        Returns 119 features:
        - Immediate danger (3): danger 1 cell ahead straight/right/left
        - Lookahead danger (3): danger 2 cells ahead straight/right/left
        - Current direction (4): one-hot encoding
        - Food direction (4): relative position to head
        - Full grid (100): spatial awareness (head=1.5, body=1.0, tail=0.5, food=2.0)
        - Snake length (1): normalized length
        - Accessible space (1): normalized reachable cells
        - Can reach tail (1): binary flag
        - Path length to food (1): normalized BFS distance
        - Space after eating (1): normalized accessible space if food is eaten

        Enhanced features help the AI plan ahead and avoid trapping itself.
        """
        head = self.snake_body[0]

        # === MULTI-STEP DANGER DETECTION (helps avoid trapping) ===
        # Current direction vector
        dir_vec = self.dirs[self.direction]

        # Points straight ahead (1 and 2 cells)
        point_straight_1 = [head[0] + dir_vec[0], head[1] + dir_vec[1]]
        point_straight_2 = [head[0] + dir_vec[0]*2, head[1] + dir_vec[1]*2]

        # Points to the right (clockwise turn)
        if self.direction == 0:  # UP -> RIGHT
            point_right_1 = [head[0] + 1, head[1]]
            point_right_2 = [head[0] + 2, head[1]]
        elif self.direction == 1:  # DOWN -> LEFT
            point_right_1 = [head[0] - 1, head[1]]
            point_right_2 = [head[0] - 2, head[1]]
        elif self.direction == 2:  # LEFT -> UP
            point_right_1 = [head[0], head[1] - 1]
            point_right_2 = [head[0], head[1] - 2]
        else:  # RIGHT -> DOWN
            point_right_1 = [head[0], head[1] + 1]
            point_right_2 = [head[0], head[1] + 2]

        # Points to the left (counter-clockwise turn)
        if self.direction == 0:  # UP -> LEFT
            point_left_1 = [head[0] - 1, head[1]]
            point_left_2 = [head[0] - 2, head[1]]
        elif self.direction == 1:  # DOWN -> RIGHT
            point_left_1 = [head[0] + 1, head[1]]
            point_left_2 = [head[0] + 2, head[1]]
        elif self.direction == 2:  # LEFT -> DOWN
            point_left_1 = [head[0], head[1] + 1]
            point_left_2 = [head[0], head[1] + 2]
        else:  # RIGHT -> UP
            point_left_1 = [head[0], head[1] - 1]
            point_left_2 = [head[0], head[1] - 2]

        # Immediate danger (1 cell ahead) - excludes tail
        danger_straight_1 = 1.0 if self._is_danger(point_straight_1) else 0.0
        danger_right_1 = 1.0 if self._is_danger(point_right_1) else 0.0
        danger_left_1 = 1.0 if self._is_danger(point_left_1) else 0.0

        # Lookahead danger (2 cells ahead) - excludes tail AND second-to-last segment
        danger_straight_2 = 1.0 if self._is_danger_lookahead(point_straight_2) else 0.0
        danger_right_2 = 1.0 if self._is_danger_lookahead(point_right_2) else 0.0
        danger_left_2 = 1.0 if self._is_danger_lookahead(point_left_2) else 0.0

        # === CURRENT DIRECTION (one-hot) ===
        dir_up = 1.0 if self.direction == 0 else 0.0
        dir_down = 1.0 if self.direction == 1 else 0.0
        dir_left = 1.0 if self.direction == 2 else 0.0
        dir_right = 1.0 if self.direction == 3 else 0.0

        # === FOOD DIRECTION (relative to head) ===
        if self.food_pos is not None:
            food_up = 1.0 if self.food_pos[1] < head[1] else 0.0
            food_down = 1.0 if self.food_pos[1] > head[1] else 0.0
            food_left = 1.0 if self.food_pos[0] < head[0] else 0.0
            food_right = 1.0 if self.food_pos[0] > head[0] else 0.0
        else:
            # No food available (grid is full - game won!)
            food_up = food_down = food_left = food_right = 0.0

        # === FULL GRID (complete spatial awareness with differentiated parts) ===
        grid = np.zeros(self.grid_width * self.grid_height, dtype=np.float32)

        # Mark snake body segments with different values
        # Head = 1.5 (brightest - current position)
        # Body = 1.0 (middle segments)
        # Tail = 0.5 (dimmest - will move away soon)
        for i, segment in enumerate(self.snake_body):
            idx = segment[1] * self.grid_width + segment[0]
            if i == 0:  # Head
                grid[idx] = 1.5
            elif i == len(self.snake_body) - 1:  # Tail
                grid[idx] = 0.5
            else:  # Body
                grid[idx] = 1.0

        # Mark food position
        if self.food_pos is not None:
            idx = self.food_pos[1] * self.grid_width + self.food_pos[0]
            grid[idx] = 2.0

        # === SNAKE LENGTH (normalized) ===
        max_length = self.grid_width * self.grid_height
        normalized_length = len(self.snake_body) / max_length

        # === NEW FEATURES FOR BETTER SPATIAL AWARENESS ===

        # 1. Accessible space (normalized)
        accessible_space = self._count_accessible_space()
        normalized_accessible = accessible_space / max_length

        # 2. Can reach tail (binary)
        can_reach_tail = 1.0 if self._can_reach_tail() else 0.0

        # 3. Path length to food (normalized)
        path_to_food = self._bfs_path_length(head, self.food_pos, exclude_tail=True)
        # Normalize: -1 (no path) -> 0.0, 0-20 steps -> 0.0-1.0
        if path_to_food == -1:
            normalized_path_to_food = 0.0  # No path
        else:
            # Normalize by grid diagonal (max possible distance)
            max_distance = self.grid_width + self.grid_height
            normalized_path_to_food = min(1.0, path_to_food / max_distance)

        # 4. Space after eating food (normalized)
        space_after_eating = self._simulate_eating_food()
        normalized_space_after = space_after_eating / max_length

        # === HAMILTON PATH FEATURES (if enabled) ===
        if self.hamilton_planner is not None:
            # Get optimal Hamilton direction
            hamilton_dir = self.hamilton_planner.get_next_direction(head, self.direction)

            # One-hot encoding of Hamilton direction
            ham_up = 1.0 if hamilton_dir == 0 else 0.0
            ham_down = 1.0 if hamilton_dir == 1 else 0.0
            ham_left = 1.0 if hamilton_dir == 2 else 0.0
            ham_right = 1.0 if hamilton_dir == 3 else 0.0

            # Should follow Hamilton? (True when snake is large)
            should_follow = 1.0 if self.hamilton_planner.should_follow_hamilton(
                len(self.snake_body), self.grid_width
            ) else 0.0
        else:
            # No Hamilton guidance
            ham_up = ham_down = ham_left = ham_right = 0.0
            should_follow = 0.0

        # Combine all features
        state = np.concatenate([
            [danger_straight_1, danger_right_1, danger_left_1],  # Immediate danger (3)
            [danger_straight_2, danger_right_2, danger_left_2],  # Lookahead danger (3)
            [dir_up, dir_down, dir_left, dir_right],              # Direction (4)
            [food_up, food_down, food_left, food_right],          # Food direction (4)
            grid,                                                  # Grid (100)
            [normalized_length],                                   # Snake length (1)
            [normalized_accessible],                               # Accessible space (1)
            [can_reach_tail],                                      # Can reach tail (1)
            [normalized_path_to_food],                             # Path to food (1)
            [normalized_space_after],                              # Space after eating (1)
            [ham_up, ham_down, ham_left, ham_right],              # Hamilton direction (4)
            [should_follow]                                        # Should follow Hamilton (1)
        ])

        return state

    def step(self, action):
        """
        Take action and return (state, reward, done)

        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        Returns:
            state: new state
            reward: reward for this action
            done: whether game is over
        """
        self.steps += 1

        # Save previous body for smooth rendering
        self.prev_body = [seg[:] for seg in self.snake_body]

        # Calculate accessible space BEFORE move (for space awareness penalty)
        old_accessible_space = self._count_accessible_space()

        # Update direction (prevent 180-degree turns)
        new_direction = action
        current_vec = self.dirs[self.direction]
        new_vec = self.dirs[new_direction]

        # Check if opposite direction
        if not (current_vec[0] == -new_vec[0] and current_vec[1] == -new_vec[1]):
            self.direction = new_direction

        # Move snake
        dx, dy = self.dirs[self.direction]
        head_x, head_y = self.snake_body[0]
        new_head = [head_x + dx, head_y + dy]

        # Reset accumulator for smooth animation
        if self.render_mode:
            self.accum = 0.0

        # Check collision - Different penalties for different collision types
        collision_type = self._check_collision_type(new_head)
        if collision_type:
            self.game_over = True
            if collision_type == 'wall':
                reward = -100  # Heavy penalty for hitting wall
            elif collision_type == 'self':
                reward = -100  # Heavy penalty for hitting itself
            return self._get_state(), reward, True

        self.snake_body.insert(0, new_head)

        # Check if food eaten
        if self.food_pos and new_head == self.food_pos:
            self.score += 1
            reward = 100  # Large reward for eating food (increased from 10 to 100)
            self.food_pos = self._spawn_food()

            # Hamilton bonus: extra reward if we're following the path at high scores
            if self.hamilton_planner is not None:
                if self.hamilton_planner.should_follow_hamilton(len(self.snake_body), self.grid_width):
                    optimal_action = self.hamilton_planner.get_next_direction(new_head, self.direction)
                    if action == optimal_action:
                        reward += 20  # Bonus for following Hamilton when we should
        else:
            self.snake_body.pop()

            # NEW REWARD STRUCTURE: Focus on space preservation, not greedy food chasing
            # Calculate accessible space AFTER move
            new_accessible_space = self._count_accessible_space()

            # Base reward for surviving
            reward = 0.0

            # Space awareness: Penalize reducing accessible space
            space_reduction = old_accessible_space - new_accessible_space
            if space_reduction > 0:
                # Losing space is bad - penalize proportionally
                reward -= space_reduction * 0.5
            elif space_reduction < 0:
                # Gaining space is good (though rare without eating)
                reward += abs(space_reduction) * 0.2

            # Small bonus for maintaining high space availability
            snake_length = len(self.snake_body)
            space_ratio = new_accessible_space / max(1, snake_length)
            if space_ratio >= 2.0:
                reward += 0.5  # Plenty of room
            elif space_ratio < 1.5:
                reward -= 1.0  # Getting cramped

            # Tiny directional hint (much smaller than before)
            # Only give this hint when there's plenty of space
            if new_accessible_space > snake_length * 2 and self.food_pos is not None:
                head = self.snake_body[0]
                distance = abs(head[0] - self.food_pos[0]) + abs(head[1] - self.food_pos[1])
                # Store previous distance if not exists
                if not hasattr(self, 'prev_food_distance'):
                    self.prev_food_distance = distance

                if distance < self.prev_food_distance:
                    reward += 0.1  # Tiny bonus for moving toward food

                self.prev_food_distance = distance

        # REMOVED: Timeout penalty - we want the snake to play carefully at high scores
        # The snake should take as long as it needs

        return self._get_state(), reward, False

    def render_frame(self):
        """Render the current game state with smooth interpolation (only if render_mode=True)"""
        if not self.render_mode:
            return

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle window resize
            if event.type == pygame.VIDEORESIZE:
                if not self.is_fullscreen:
                    self._recalculate_display(event.w, event.h)
                    self.game_window = pygame.display.set_mode((self.frame_width, self.frame_height), pygame.RESIZABLE)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_h:
                    # Toggle Hamilton path visibility
                    self.show_hamilton_path = not self.show_hamilton_path
                elif event.key == pygame.K_F11:
                    # Toggle fullscreen
                    self.is_fullscreen = not self.is_fullscreen
                    if self.is_fullscreen:
                        display_info = pygame.display.Info()
                        screen_width = display_info.current_w
                        screen_height = display_info.current_h
                        self._recalculate_display(screen_width, screen_height)
                        self.game_window = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
                    else:
                        self.game_window = pygame.display.set_mode((self.frame_width, self.frame_height), pygame.RESIZABLE)

        # Update accumulator for smooth movement
        dt = self.fps_controller.tick(self.fps) / 1000.0
        if not self.game_over:
            self.accum += dt

        # Clear screen
        self.game_window.fill(self.black)

        # Draw Hamilton path as subtle white line in background (if enabled)
        if self.show_hamilton_path and self.hamilton_planner is not None:
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    current_idx = self.hamilton_planner.path_map[y, x]
                    next_idx = (current_idx + 1) % (self.grid_width * self.grid_height)

                    # Find next cell in path
                    for ny in range(self.grid_height):
                        for nx in range(self.grid_width):
                            if self.hamilton_planner.path_map[ny, nx] == next_idx:
                                # Draw thin white line from current to next
                                start = (x * self.cell_size + self.cell_size // 2,
                                        y * self.cell_size + self.cell_size // 2)
                                end = (nx * self.cell_size + self.cell_size // 2,
                                      ny * self.cell_size + self.cell_size // 2)
                                pygame.draw.line(self.game_window, (255, 255, 255), start, end, 2)
                                break
                        else:
                            continue
                        break

        # Calculate interpolation factor (alpha) for smooth movement
        alpha = 0.0 if self.game_over else max(0.0, min(self.accum / self.step_time, 1.0))

        # Ensure prev_body matches current body length
        if len(self.prev_body) < len(self.snake_body):
            self.prev_body = self.prev_body + [self.prev_body[-1][:]] * (len(self.snake_body) - len(self.prev_body))

        # Calculate interpolated positions for smooth movement
        interp_centers = []
        for prev_seg, cur_seg in zip(self.prev_body, self.snake_body):
            pa = cell_center(prev_seg, self.cell_size)
            pb = cell_center(cur_seg, self.cell_size)
            interp_centers.append(lerp(pa, pb, alpha))

        # Insert corner points so elbows are filled boxes (smooth corners)
        tube_points = [interp_centers[0]]
        for i in range(1, len(interp_centers)):
            corner = cell_center(self.prev_body[i - 1], self.cell_size)
            if (abs(tube_points[-1][0] - corner[0]) > 0.01) or (abs(tube_points[-1][1] - corner[1]) > 0.01):
                tube_points.append(corner)
            tube_points.append(interp_centers[i])

        # Draw snake with filled corners (same as original snake.py)
        draw_snake_with_filled_corners(self.game_window, tube_points, self.snake_thickness, self.green)

        # Draw food
        if self.food_pos:
            food_size = int(self.cell_size * 0.98)
            offset = (self.cell_size - food_size) // 2
            if self.food_pos is not None:
                pygame.draw.rect(
                    self.game_window,
                    self.red,
                    pygame.Rect(self.food_pos[0] * self.cell_size + offset,
                               self.food_pos[1] * self.cell_size + offset,
                               food_size, food_size)
                )

        # Draw score
        score_text = self.score_font.render(f'Score: {self.score}', True, self.red)
        self.game_window.blit(score_text, (10, 10))

        # Update display
        pygame.display.update()

    def close(self):
        """Clean up pygame resources"""
        if self.render_mode:
            pygame.quit()
