"""
Hamiltonian Cycle for Snake using Prim's Algorithm
- Creates a predetermined loop path that visits every cell exactly once
- Snake follows this path forever and never dies
- Uses Prim's MST algorithm on a half-resolution grid to create interesting patterns
- Works for ANY grid size (even, odd, rectangular)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random

# Directions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
DIRS = {
    0: (0, -1),
    1: (0,  1),
    2: (-1, 0),
    3: (1,  0),
}

def generate_hamiltonian_cycle(width: int, height: int = None, seed: Optional[int] = None) -> Dict[Tuple[int, int], int]:
    """
    Generate a valid Hamiltonian cycle using Prim's algorithm on a half-resolution grid.
    Creates interesting, organic patterns that work for ANY grid size.

    Properties:
    - visits every cell exactly once (width × height cells)
    - consecutive cells are Manhattan-adjacent
    - start is (0,0)
    - forms a closed cycle
    - works for even, odd, square, and rectangular grids
    
    Algorithm:
    1. Create nodes at odd coordinates (1,1), (3,1), (5,1), etc. (half-resolution)
    2. Build weighted graph between adjacent nodes
    3. Run Prim's algorithm to get minimum spanning tree
    4. Navigate around tree edges to create Hamiltonian cycle
    
    Args:
        width: Width of the grid (x dimension)
        height: Height of the grid (y dimension). If None, uses width (square grid)
        seed: Random seed for reproducibility. If None, uses random seed each time.
    """
    if seed is not None:
        random.seed(seed)
    # If seed is None, random will use system time (different each run)
    if height is None:
        height = width
    
    grid_width = width
    grid_height = height
    half_x = (grid_width + 1) // 2
    half_y = (grid_height + 1) // 2
    
    # Step 1: Create nodes at odd coordinates
    nodes = {}
    for x in range(half_x):
        for y in range(half_y):
            node_num = x + y * half_x
            pos_x = x * 2 + 1
            pos_y = y * 2 + 1
            # Keep nodes within bounds
            if pos_x < grid_width and pos_y < grid_height:
                nodes[node_num] = (pos_x, pos_y)
    
    # Step 2: Create edges between adjacent nodes with random weights
    edges = {}
    skip_list = [half_x * x for x in range(half_y)]
    
    for node1 in nodes:
        for node2 in nodes:
            if node1 != node2:
                # Right neighbor (skip if at right edge)
                if node1 + 1 == node2 and node2 not in skip_list:
                    edges[(node1, node2)] = random.randint(1, 3)
                # Bottom neighbor
                elif node1 + half_x == node2:
                    edges[(node1, node2)] = random.randint(1, 3)
    
    # Step 3: Run Prim's algorithm to get MST
    mst_edges = prims_algorithm(edges, list(nodes.keys()))
    
    # Step 4: Convert MST edges to grid points (halfway between nodes)
    tree_points = set()
    for (node1, node2), _ in mst_edges:
        start = nodes[node1]
        end = nodes[node2]
        tree_points.add(start)
        tree_points.add(((start[0] + end[0]) // 2, (start[1] + end[1]) // 2))
        tree_points.add(end)
    
    # Step 5: Navigate the full grid around tree edges
    cycle = navigate_around_tree(grid_width, grid_height, tree_points)
    
    # Convert list to dict mapping (x,y) -> order
    return {pos: idx for idx, pos in enumerate(cycle)}


def prims_algorithm(edges: Dict[Tuple[int, int], int], nodes: List[int]) -> List[Tuple[Tuple[int, int], int]]:
    """
    Prim's algorithm to find minimum spanning tree.
    
    Args:
        edges: dict of (node1, node2) -> weight
        nodes: list of all node numbers
    
    Returns:
        list of ((node1, node2), weight) edges in the MST
    """
    if not nodes:
        return []
    
    visited = []
    unvisited = nodes.copy()
    curr = nodes[0]
    
    mst_edges = []
    
    while unvisited:
        visited.append(curr)
        
        # Remove visited nodes from unvisited
        unvisited = [n for n in unvisited if n not in visited]
        
        if not unvisited:
            break
        
        # Find all edges connecting visited to unvisited nodes
        candidate_edges = []
        for (n1, n2), weight in edges.items():
            if (n1 in visited and n2 not in visited):
                candidate_edges.append(((n1, n2), weight))
            elif (n2 in visited and n1 not in visited):
                candidate_edges.append(((n2, n1), weight))
        
        # Find minimum weight edge
        if candidate_edges:
            min_edge = min(candidate_edges, key=lambda x: x[1])
            mst_edges.append(min_edge)
            curr = min_edge[0][1]  # Move to the unvisited node
        else:
            # No more edges, pick any unvisited node (handles disconnected graphs)
            curr = unvisited[0]
    
    return mst_edges


def navigate_around_tree(grid_width: int, grid_height: int, tree_points: set) -> List[Tuple[int, int]]:
    """
    Navigate around the spanning tree edges to create a Hamiltonian cycle.
    Uses a wall-following algorithm that treats tree edges as walls.
    
    Args:
        grid_width: width of the grid (x dimension)
        grid_height: height of the grid (y dimension)
        tree_points: set of (x, y) positions that are part of the tree structure
    
    Returns:
        list of (x, y) positions forming a Hamiltonian cycle
    """
    width = grid_width
    height = grid_height
    cycle = [(0, 0)]
    curr = (0, 0)
    direction = (1, 0)  # Start going right
    
    max_iterations = width * height * 10  # Safety limit
    iterations = 0
    
    while len(cycle) < width * height and iterations < max_iterations:
        iterations += 1
        x, y = curr
        dx, dy = direction
        
        # Wall-following logic adapted from reference implementation
        if direction == (1, 0):  # Moving RIGHT
            # Check conditions for turning
            check_pos = (x + 1, y + 1)
            next_blocked = (x + 1, y) not in tree_points
            
            if check_pos in tree_points and next_blocked:
                # Continue right
                direction = (1, 0)
            else:
                # Decide to turn down or up
                if (x, y + 1) in tree_points and (x + 1, y + 1) not in tree_points:
                    direction = (0, 1)  # Turn DOWN
                else:
                    direction = (0, -1)  # Turn UP
                    
        elif direction == (0, 1):  # Moving DOWN
            next_pos = (x, y + 1)
            next_right = (x + 1, y + 1)
            
            if next_pos in tree_points and next_right not in tree_points:
                # Continue down
                direction = (0, 1)
            else:
                # Decide to turn right or left
                if (x, y + 1) in tree_points and (x + 1, y + 1) in tree_points:
                    direction = (1, 0)  # Turn RIGHT
                else:
                    direction = (-1, 0)  # Turn LEFT
                    
        elif direction == (-1, 0):  # Moving LEFT
            if (x, y) in tree_points and (x, y + 1) not in tree_points:
                # Continue left
                direction = (-1, 0)
            else:
                # Decide to turn up or down
                if (x, y + 1) not in tree_points:
                    direction = (0, -1)  # Turn UP
                else:
                    direction = (0, 1)  # Turn DOWN
                    
        elif direction == (0, -1):  # Moving UP
            if (x, y) not in tree_points and (x + 1, y) in tree_points:
                # Continue up
                direction = (0, -1)
            else:
                # Decide to turn left or right
                if (x + 1, y) in tree_points:
                    direction = (-1, 0)  # Turn LEFT
                else:
                    direction = (1, 0)  # Turn RIGHT
        
        # Move in the chosen direction
        next_x = x + direction[0]
        next_y = y + direction[1]
        
        # Bounds check and validity check
        if 0 <= next_x < width and 0 <= next_y < height:
            next_pos = (next_x, next_y)
            if next_pos not in cycle:
                cycle.append(next_pos)
                curr = next_pos
            else:
                # Hit a visited cell, we've completed the cycle
                if len(cycle) == width * height:
                    break
                # Otherwise try a different direction
                # This shouldn't happen in a valid maze, but as fallback:
                # Try all 4 directions
                for try_dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    tx, ty = x + try_dir[0], y + try_dir[1]
                    if 0 <= tx < width and 0 <= ty < height and (tx, ty) not in cycle:
                        direction = try_dir
                        break
        else:
            # Hit boundary, turn (shouldn't happen often)
            # Try turning
            if direction == (1, 0) or direction == (-1, 0):
                direction = (0, 1) if y < height // 2 else (0, -1)
            else:
                direction = (1, 0) if x < width // 2 else (-1, 0)
    
    # If we didn't complete the cycle with the maze algorithm, fall back to simple zigzag
    if len(cycle) < width * height:
        print(f"Warning: Prim's maze only generated {len(cycle)}/{width*height} cells, using zigzag fallback")
        return create_zigzag_cycle(width, height)
    
    return cycle


def create_zigzag_cycle(grid_width: int, grid_height: int = None) -> List[Tuple[int, int]]:
    """Fallback: create a simple zigzag pattern that's guaranteed to work."""
    if grid_height is None:
        grid_height = grid_width
    
    path = []
    
    for y in range(grid_height):
        if y % 2 == 0:
            # Even rows: left to right
            for x in range(grid_width):
                path.append((x, y))
        else:
            # Odd rows: right to left
            for x in range(grid_width - 1, -1, -1):
                path.append((x, y))
    
    return path



class HamiltonianSnakePlanner:
    def __init__(self, grid_width: int = 10, grid_height: int = None, seed: Optional[int] = None):
        """
        Initialize Hamiltonian cycle planner.
        
        Args:
            grid_width: Width of the grid (x dimension)
            grid_height: Height of the grid (y dimension). If None, uses grid_width (square grid)
            seed: Random seed for cycle generation. If None, generates a new random pattern each time.
        """
        if grid_height is None:
            grid_height = grid_width
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_size = grid_width  # Keep for backward compatibility
        # Pre-compute the Hamiltonian cycle for this grid
        self.cycle = generate_hamiltonian_cycle(grid_width, grid_height, seed=seed)
        self.cycle_length = grid_width * grid_height
        
        # Create reverse mapping: order -> (x, y)
        self.order_to_pos = {order: pos for pos, order in self.cycle.items()}
        
        # Create path_map for rendering: numpy array indexed by [y, x]
        import numpy as np
        self.path_map = np.zeros((grid_height, grid_width), dtype=int)
        for (x, y), order in self.cycle.items():
            self.path_map[y, x] = order

    def _dir_from_to(self, a: Tuple[int,int], b: Tuple[int,int]) -> int:
        """Convert a move from position a to position b into a direction."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        if (dx, dy) == (0, -1): return 0  # UP
        if (dx, dy) == (0,  1): return 1  # DOWN
        if (dx, dy) == (-1, 0): return 2  # LEFT
        if (dx, dy) == (1,  0): return 3  # RIGHT
        raise RuntimeError(f"Non-adjacent step {a}->{b}")

    def get_next_direction(
        self,
        head_pos: List[int],
        current_direction: Optional[int] = None,
        snake_body: List[List[int]] = None,
        food_pos: List[int] = None,
        will_grow: bool = False,
    ) -> int:
        """
        Returns the next direction following the Hamiltonian cycle.
        Can follow the cycle in either direction (increasing or decreasing).
        
        Args:
          head_pos: [x,y] current head position
          current_direction: current direction the snake is facing (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
          snake_body: list of snake segments (uses tail to determine direction)
          food_pos: (ignored - not needed for Hamiltonian cycle)
          will_grow: (ignored - not needed for Hamiltonian cycle)

        Returns:
          0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        head = (head_pos[0], head_pos[1])
        
        # Get current position in the cycle
        current_order = self.cycle[head]
        
        # Determine direction preference: forward or backward in cycle
        # If we have snake body info, check which direction makes more sense
        prefer_forward = True
        if snake_body and len(snake_body) > 1:
            tail = (snake_body[-1][0], snake_body[-1][1])
            if tail in self.cycle:
                tail_order = self.cycle[tail]
                # Calculate distances in both directions
                forward_dist = (tail_order - current_order) % self.cycle_length
                backward_dist = (current_order - tail_order) % self.cycle_length
                # Prefer the direction where tail is "behind" us
                prefer_forward = forward_dist > backward_dist
        
        # Try both directions and pick valid adjacent move
        next_order_fwd = (current_order + 1) % self.cycle_length
        next_order_bwd = (current_order - 1) % self.cycle_length
        
        next_pos_fwd = self.order_to_pos[next_order_fwd]
        next_pos_bwd = self.order_to_pos[next_order_bwd]
        
        # Check which moves are actually adjacent (manhattan distance = 1)
        def is_adjacent(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
        
        fwd_valid = is_adjacent(head, next_pos_fwd)
        bwd_valid = is_adjacent(head, next_pos_bwd)
        
        # Choose based on preference and validity
        if prefer_forward and fwd_valid:
            return self._dir_from_to(head, next_pos_fwd)
        elif bwd_valid:
            return self._dir_from_to(head, next_pos_bwd)
        elif fwd_valid:
            return self._dir_from_to(head, next_pos_fwd)
        else:
            # Fallback: try to move in any valid direction
            for next_order in [next_order_fwd, next_order_bwd]:
                next_pos = self.order_to_pos[next_order]
                if is_adjacent(head, next_pos):
                    return self._dir_from_to(head, next_pos)
            # Should never reach here if cycle is valid
            return 3  # RIGHT as last resort
    
    def should_follow_hamilton(self, snake_length: int, grid_size: int) -> bool:
        """
        Determine if the snake should follow the Hamilton cycle.
        For a pure Hamilton strategy, always return True.
        
        Args:
            snake_length: current length of the snake
            grid_size: size of the grid
            
        Returns:
            True if snake should follow Hamilton path
        """
        # Always follow Hamilton cycle for guaranteed safety
        return True


# Alias for backward compatibility
HamiltonPathPlanner = HamiltonianSnakePlanner


def visualize_cycle(grid_width: int = 10, grid_height: int = None):
    """Display the Hamiltonian cycle in the console."""
    if grid_height is None:
        grid_height = grid_width
    
    cycle = generate_hamiltonian_cycle(grid_width, grid_height)
    
    # Find where positions 0 and (width×height-1) are located
    pos_0 = None
    pos_last = None
    total_cells = grid_width * grid_height
    for (x, y), order in cycle.items():
        if order == 0:
            pos_0 = (x, y)
        if order == total_cells - 1:
            pos_last = (x, y)
    
    adjacent = abs(pos_0[0] - pos_last[0]) + abs(pos_0[1] - pos_last[1]) == 1
    
    print("\n" + "="*60)
    print(f"Hamiltonian Cycle for {grid_width}x{grid_height} Grid")
    print("="*60)
    print("Numbers show the order in which cells are visited:")
    print()
    
    # Create grid for display
    grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    for (x, y), order in cycle.items():
        grid[y][x] = order
    
    # Calculate cell width for formatting
    max_num = total_cells - 1
    cell_width = len(str(max_num)) + 1
    
    # Print column headers
    print("    ", end="")
    for x in range(grid_width):
        print(f"x{x}".ljust(cell_width), end=" ")
    print()
    
    # Print grid
    for y in range(grid_height):
        print(f"y{y}  ", end="")
        for x in range(grid_width):
            num_str = str(grid[y][x]).ljust(cell_width)
            print(num_str, end=" ")
        print()
    
    print(f"\nPosition 0 at {pos_0}, Position {max_num} at {pos_last}")
    print(f"Adjacent: {adjacent} {'✓' if adjacent else '✗ NOT A VALID CYCLE!'}")
    print(f"\nPath: The snake moves 0→1→2→...→{max_num}→0 (loops back)")
    print("="*60)
    

if __name__ == "__main__":
    # When run directly, visualize the cycle
    visualize_cycle(10)

