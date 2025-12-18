"""
Watch the trained Snake AI play
Loads a trained model and displays the game
"""

import time
from game.environment import SnakeEnv
from training.snake_ai import DQNAgent


def watch_agent_play(
    model_path='snake_model.pth',
    num_games=10,
    fps=60,
    speed_cells=6,
    grid_width=10,
    grid_height=10,
    delay_between_games=2.0,
    show_hamilton_path=False
):
    """
    Watch the trained agent play Snake

    Args:
        model_path: Path to the trained model
        num_games: Number of games to play
        fps: Frames per second for rendering (60 for smooth animation)
        speed_cells: Movement speed in cells per second (lower = slower, easier to watch)
        grid_width: Width of the grid
        grid_height: Height of the grid
        delay_between_games: Seconds to wait between games
        show_hamilton_path: If True, show Hamilton path overlay
    """
    # Create environment with rendering enabled (cell_size auto-calculated)
    env = SnakeEnv(render=True, grid_width=grid_width, grid_height=grid_height,
                   cell_size=None, fps=fps, speed_cells=speed_cells)

    # Create agent
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_space,
        grid_width=env.grid_width,
        grid_height=env.grid_height
    )

    # Load trained model
    try:
        agent.load(model_path)
        print(f"Model loaded from {model_path}")
        print(f"Model trained for {agent.n_games} episodes")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first by running: python train.py")
        return

    print("\n" + "="*60)
    print("Watching AI Play Snake")
    print("="*60)
    print(f"Number of games: {num_games}")
    print(f"FPS: {fps} | Speed: {speed_cells} cells/sec")
    print(f"Hamilton path overlay: {'ON' if show_hamilton_path else 'OFF'}")
    print("Press ESC to exit")
    print("="*60 + "\n")

    scores = []

    # Set the Hamilton path visibility on the environment
    env.show_hamilton_path = show_hamilton_path

    for game in range(num_games):
        state = env.reset()
        done = False
        steps = 0

        print(f"Game {game + 1}/{num_games} starting...")

        while not done:
            # SAFETY MECHANISM: Get safe actions before choosing
            safe_actions = env._get_safe_actions()

            # Get action from agent (no exploration, use best action, with safety)
            action = agent.get_action(state, training=False, safe_actions=safe_actions)

            # Take action (this updates game state)
            next_state, reward, done = env.step(action)

            # Render smooth animation for this move
            # Show multiple frames to display smooth movement between grid cells
            frames_per_move = max(1, int(fps / env.speed_cells))
            for _ in range(frames_per_move):
                env.render_frame()

            state = next_state
            steps += 1

        # Game over
        score = env.score
        scores.append(score)

        print(f"Game {game + 1} finished | Score: {score} | Steps: {steps}")

        # Show game over state briefly
        for _ in range(int(fps * delay_between_games)):
            env.render_frame()

    # Summary
    print("\n" + "="*60)
    print("Session Complete!")
    print("="*60)
    print(f"Games Played: {num_games}")
    print(f"Average Score: {sum(scores)/len(scores):.1f}")
    print(f"Best Score: {max(scores)}")
    print(f"Worst Score: {min(scores)}")
    print("="*60 + "\n")

    env.close()


if __name__ == "__main__":
    # Watch the AI play 10 games
    # Adjust speed_cells to control speed (lower = slower, easier to watch)
    # Keep fps=60 for smooth animation
    # Set show_hamilton_path=True to see the Hamilton cycle path as a white line
    watch_agent_play(
        model_path='snake_model.pth',
        num_games=10,
        fps=60,  # Keep at 60 for smooth animation
        speed_cells=6,  # Slower than original (6) for easier watching
        delay_between_games=2.0,
        show_hamilton_path=False  # Set to True to see Hamilton path
    )
