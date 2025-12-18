"""
Snake AI Game - Main Entry Point
Run this file to access all game modes and features
"""

import sys
import os


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print the game banner"""
    print("\n" + "="*60)
    print("  🐍  SNAKE AI GAME  🐍")
    print("="*60)


def print_menu():
    """Print the main menu"""
    print("\nChoose an option:")
    print("  1. 🎮 Play Manually (You control the snake)")
    print("  2. 🤖 Watch Hamilton Cycle Demo (Perfect AI play)")
    print("  3. 🧠 Train Neural Network (Deep Q-Learning)")
    print("  4. 👀 Watch Trained Model Play")
    print("  5. 🚪 Exit")
    print()


def get_config(mode_name, show_render=True, show_speed=True, show_grid=True):
    """
    Get configuration from user for a specific mode

    Args:
        mode_name: Name of the mode (for display)
        show_render: Whether to ask about rendering
        show_speed: Whether to ask about speed
        show_grid: Whether to ask about grid dimensions

    Returns:
        dict: Configuration dictionary with keys: render, speed, grid_width, grid_height
    """
    print("\n" + "="*60)
    print(f"Configuration for {mode_name}")
    print("="*60)
    print("Press Enter to use default values shown in [brackets]\n")

    config = {}

    # Grid dimensions
    if show_grid:
        try:
            grid_width = input("Grid width [10]: ").strip()
            config['grid_width'] = int(grid_width) if grid_width else 10

            grid_height = input("Grid height [10]: ").strip()
            config['grid_height'] = int(grid_height) if grid_height else 10
        except ValueError:
            print("Invalid input. Using default 10x10 grid.")
            config['grid_width'] = 10
            config['grid_height'] = 10
    else:
        config['grid_width'] = 10
        config['grid_height'] = 10

    # Render option
    if show_render:
        render_input = input("Enable rendering (show game window)? [Y/n]: ").strip().lower()
        config['render'] = render_input != 'n'
    else:
        config['render'] = True

    # Speed
    if show_speed:
        try:
            speed_input = input("Speed in cells/second [6]: ").strip()
            config['speed'] = int(speed_input) if speed_input else 6
        except ValueError:
            print("Invalid input. Using default speed 6.")
            config['speed'] = 6
    else:
        config['speed'] = 6

    print("\n" + "="*60)
    print("Configuration Summary:")
    print("="*60)
    print(f"  Grid Size: {config['grid_width']}x{config['grid_height']}")
    if show_render:
        print(f"  Rendering: {'ON' if config['render'] else 'OFF'}")
    if show_speed:
        print(f"  Speed: {config['speed']} cells/second")
    print("="*60 + "\n")

    return config


def manual_play():
    """Launch manual play mode"""
    print("\n" + "="*60)
    print("Starting Manual Play Mode...")
    print("="*60 + "\n")

    # Get configuration
    config = get_config("Manual Play", show_render=False, show_speed=True, show_grid=True)

    print("\n" + "="*60)
    print("Controls:")
    print("  Arrow Keys (or WASD) - Move the snake")
    print("  R or SPACE - Restart game")
    print("  H - Toggle Hamilton path overlay")
    print("  F11 - Toggle fullscreen")
    print("  ESC - Exit")
    print("  You can also resize the window by dragging!")
    print("="*60 + "\n")

    try:
        from game.manual_play import play_game
        play_game(
            grid_width=config['grid_width'],
            grid_height=config['grid_height'],
            speed_cells=config['speed']
        )
    except ImportError as e:
        print(f"❌ Error: Could not import manual play module: {e}")
        print("Make sure game/manual_play.py exists.")
        input("\nPress Enter to return to menu...")
    except Exception as e:
        print(f"❌ Error during manual play: {e}")
        input("\nPress Enter to return to menu...")


def hamilton_demo():
    """Launch Hamilton cycle demonstration"""
    print("\n" + "="*60)
    print("Starting Hamilton Cycle Demo...")
    print("The snake will follow a perfect path that visits every cell")
    print("="*60 + "\n")

    # Get configuration
    config = get_config("Hamilton Demo", show_render=False, show_speed=True, show_grid=True)

    try:
        num_games = int(input("\nNumber of games to watch [1]: ").strip() or "1")
    except ValueError:
        num_games = 1

    show_path = input("Show Hamilton path overlay? [Y/n]: ").strip().lower() != 'n'

    print("\n" + "="*60)
    print("Press H to toggle path visualization | Press ESC to exit")
    print("="*60 + "\n")

    try:
        from demos.hamilton_demo import test_hamilton_path
        test_hamilton_path(
            num_games=num_games,
            fps=60,
            speed_cells=config['speed'],
            grid_width=config['grid_width'],
            grid_height=config['grid_height'],
            delay_between_games=2.0,
            show_path_overlay=show_path
        )
    except ImportError as e:
        print(f"❌ Error: Could not import Hamilton demo: {e}")
        print("Make sure demos/hamilton_demo.py exists.")
    except Exception as e:
        print(f"❌ Error during Hamilton demo: {e}")

    input("\nPress Enter to return to menu...")


def train_network():
    """Launch neural network training"""
    print("\n" + "="*60)
    print("Starting Neural Network Training...")
    print("="*60 + "\n")

    # Get configuration
    config = get_config("Neural Network Training", show_render=True, show_speed=True, show_grid=True)

    try:
        episodes = int(input("\nNumber of episodes [1000]: ").strip() or "1000")
    except ValueError:
        episodes = 1000

    print(f"\n🚀 Starting training for {episodes} episodes...")
    if config['render']:
        print("⚠️  Rendering enabled - training will be slower but watchable")
    else:
        print("⚡ Headless mode - maximum speed")
    print()

    try:
        from training.train import train
        train(
            num_episodes=episodes,
            render=config['render'],
            render_speed=config['speed'],
            grid_width=config['grid_width'],
            grid_height=config['grid_height']
        )
    except ImportError as e:
        print(f"❌ Error: Could not import training module: {e}")
        print("Make sure training/train.py and training/snake_ai.py exist.")
    except Exception as e:
        print(f"❌ Error during training: {e}")

    input("\nPress Enter to return to menu...")


def watch_trained_model():
    """Watch a trained neural network play"""
    print("\n" + "="*60)
    print("Watch Trained Model Play...")
    print("="*60 + "\n")

    model_path = input("Model path [snake_model.pth]: ").strip() or "snake_model.pth"

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Train a model first using option 3.")
        input("\nPress Enter to return to menu...")
        return

    # Get configuration
    config = get_config("Watch Trained Model", show_render=False, show_speed=True, show_grid=True)

    try:
        num_games = int(input("\nNumber of games to watch [5]: ") or "5")
    except ValueError:
        num_games = 5

    show_path = input("Show Hamilton path overlay? [y/N]: ").strip().lower() == 'y'

    print(f"\n🎮 Watching trained model play {num_games} game(s)...")
    print("Press ESC to exit\n")

    try:
        from training.watch import watch_agent_play
        watch_agent_play(
            model_path=model_path,
            num_games=num_games,
            fps=60,
            speed_cells=config['speed'],
            grid_width=config['grid_width'],
            grid_height=config['grid_height'],
            show_hamilton_path=show_path
        )
    except ImportError as e:
        print(f"❌ Error: Could not import watch module: {e}")
        print("Make sure training/watch.py exists.")
    except ValueError:
        print("❌ Invalid input. Returning to menu.")
    except Exception as e:
        print(f"❌ Error while watching model: {e}")

    input("\nPress Enter to return to menu...")


def main():
    """Main menu loop"""
    while True:
        clear_screen()
        print_banner()
        print_menu()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            manual_play()
        elif choice == '2':
            hamilton_demo()
        elif choice == '3':
            train_network()
        elif choice == '4':
            watch_trained_model()
        elif choice == '5':
            print("\n👋 Thanks for playing! Goodbye!\n")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
        sys.exit(0)
