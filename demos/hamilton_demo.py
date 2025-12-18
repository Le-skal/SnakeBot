"""
Test Snake with Pure Hamilton Path Following
This demonstrates a guaranteed perfect strategy that can achieve 100% success
The snake follows a pre-computed Hamilton cycle that visits every cell exactly once
"""

import time
import pygame
from game.environment import SnakeEnv


def test_hamilton_path(
    num_games=5,
    fps=60,
    speed_cells=8,
    grid_width=10,
    grid_height=10,
    delay_between_games=2.0,
    show_path_overlay=True
):
    """
    Test the snake following only the Hamilton cycle path
    This should theoretically achieve perfect scores (eating all food without dying)

    Args:
        num_games: Number of games to play
        fps: Frames per second for rendering
        speed_cells: Movement speed in cells per second
        grid_width: Width of the grid
        grid_height: Height of the grid
        delay_between_games: Seconds to wait between games
        show_path_overlay: If True, display Hamilton path numbers
    """
    # Create environment with Hamilton path enabled (cell_size auto-calculated)
    env = SnakeEnv(
        render=True,
        grid_width=grid_width,
        grid_height=grid_height,
        cell_size=None,
        fps=fps,
        speed_cells=speed_cells,
        use_hamilton=True
    )

    if env.hamilton_planner is None:
        print("ERROR: Hamilton path planner not available!")
        print("Please ensure hamilton_path.py is in the same directory.")
        return

    print("\n" + "="*60)
    print("Testing Pure Hamilton Path Following")
    print("="*60)
    print(f"Number of games: {num_games}")
    print(f"FPS: {fps} | Speed: {speed_cells} cells/sec")
    print(f"Hamilton path overlay: {'ON' if show_path_overlay else 'OFF'}")
    print("Press ESC to exit")
    print("\nThe snake will follow a pre-computed zigzag path that")
    print("visits every cell exactly once, guaranteeing perfect play.")
    print("="*60 + "\n")

    scores = []

    # Set Hamilton path visibility
    env.show_hamilton_path = show_path_overlay

    for game in range(num_games):
        state = env.reset()
        done = False
        steps = 0

        print(f"Game {game + 1}/{num_games} starting...")

        while not done:
            # Get current head position and direction
            head_pos = env.snake_body[0]
            current_direction = env.direction

            # Get optimal action from Hamilton path planner
            action = env.hamilton_planner.get_next_direction(head_pos, current_direction)

            # Take action
            next_state, reward, done = env.step(action)

            # Render smooth animation for this move
            frames_per_move = max(1, int(fps / env.speed_cells))
            for _ in range(frames_per_move):
                env.render_frame()

            state = next_state
            steps += 1

            # Safety check: prevent infinite loops
            if steps > 10000:
                print(f"WARNING: Game exceeded 10000 steps. Ending game.")
                done = True

        # Game over
        score = env.score
        scores.append(score)

        print(f"Game {game + 1} finished | Score: {score} | Steps: {steps}")

        # Show game over state briefly
        for _ in range(int(fps * delay_between_games)):
            env.render_frame()

    # Summary
    print("\n" + "="*60)
    print("Hamilton Path Test Complete!")
    print("="*60)
    print(f"Games Played: {num_games}")
    print(f"Average Score: {sum(scores)/len(scores):.1f}")
    print(f"Best Score: {max(scores)}")
    print(f"Worst Score: {min(scores)}")
    print(f"Perfect Games (100% fill): {sum(1 for s in scores if s >= 97)}")
    print("="*60 + "\n")

    # Analysis
    # Max score = total cells - initial snake length (3)
    max_possible_score = env.grid_width * env.grid_height - 3
    avg_score = sum(scores) / len(scores)
    success_rate = (avg_score / max_possible_score) * 100

    print("Analysis:")
    print(f"Maximum possible score: {max_possible_score}")
    print(f"Success rate: {success_rate:.1f}%")

    if success_rate >= 99.0:
        print("✓ PERFECT! The Hamilton path achieves near-100% success.")
    elif success_rate >= 90.0:
        print("✓ EXCELLENT! The Hamilton path is very effective.")
    else:
        print("⚠ NOTE: The Hamilton path should theoretically achieve 100%.")
        print("  If seeing lower scores, there may be an implementation issue.")

    env.close()


if __name__ == "__main__":
    # Test pure Hamilton path following
    # This should achieve perfect or near-perfect scores
    test_hamilton_path(
        num_games=1,
        fps=60,
        speed_cells=20,  # Slower for easier watching
        delay_between_games=2.0,
        show_path_overlay=True  # Show the path by default
    )
