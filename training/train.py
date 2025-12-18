"""
Training script for Snake AI
Trains in headless mode (no rendering) for maximum speed
Now with interactive command-line interface!
"""

import numpy as np
from game.environment import SnakeEnv
from training.snake_ai import DQNAgent
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse
import sys
import random


def train(
    num_episodes=500,
    target_update_freq=10,
    save_freq=250,  # Save less frequently - every 250 episodes
    model_path='snake_model.pth',
    render=False,
    render_fps=60,
    render_speed=10,
    render_hamilton=False,
    grid_width=10,
    grid_height=10,
    early_stopping=True,
    patience=200,
    use_hamilton_guidance=True,
    hamilton_epsilon_start=0.95,
    hamilton_epsilon_end=0.70,  # Keep it HIGH - never go below 70%
    hamilton_epsilon_decay=0.9995  # Decay MUCH slower
):
    """
    Train the DQN agent with enhanced features and Hamilton-guided learning

    Args:
        num_episodes: Number of games to train
        target_update_freq: How often to update target network
        save_freq: How often to save the model
        model_path: Path to save the model
        render: If True, show the game while training (slower but you can watch)
        render_fps: FPS when rendering (default 60 for smooth animation)
        render_speed: Speed in cells/sec when rendering (higher = faster training)
        render_hamilton: If True, show Hamilton path as white line overlay
        grid_width: Width of the grid
        grid_height: Height of the grid
        early_stopping: If True, stop training when performance plateaus
        patience: Number of episodes to wait for improvement before stopping
        use_hamilton_guidance: If True, use Hamilton cycle as teacher policy with reward shaping
        hamilton_epsilon_start: Initial probability of following Hamilton (0.95 = 95%)
        hamilton_epsilon_end: Final probability of following Hamilton (0.70 = 70% - keeps safety net!)
        hamilton_epsilon_decay: Decay rate for Hamilton guidance (very slow)
    """
    # Create environment (with or without rendering)
    env = SnakeEnv(render=render, grid_width=grid_width, grid_height=grid_height,
                   fps=render_fps, speed_cells=render_speed)

    # Set Hamilton path visualization if requested
    env.show_hamilton_path = render_hamilton

    # Create agent with ENHANCED hyperparameters
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_space,
        hidden_size=256,  # Balanced size for efficient CNN
        lr=0.0005,  # Slightly lower learning rate for stability
        gamma=0.99,  # High discount factor - value long-term survival
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,  # Standard decay rate
        memory_size=100_000,
        batch_size=256,  # INCREASED from 64 to 256 for more stable gradients
        use_prioritized_replay=True,  # Enable prioritized experience replay
        adaptive_epsilon=True,  # Enable adaptive epsilon decay
        grid_width=env.grid_width,
        grid_height=env.grid_height
    )

    # Try to load existing model to continue training
    try:
        agent.load(model_path)
        print(f"Continuing training from episode {agent.n_games}")
    except:
        print("Starting fresh training")

    # Tracking metrics
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    best_avg_score = 0
    episodes_without_improvement = 0
    best_model_path = model_path.replace('.pth', '_best.pth')

    # Hamilton guidance tracking
    hamilton_epsilon = hamilton_epsilon_start if use_hamilton_guidance else 0.0
    hamilton_follows = 0  # Track how often Hamilton is followed

    # Recent scores for moving average
    recent_scores = deque(maxlen=100)

    print("\n" + "="*60)
    if render:
        print("Starting Training (Visual Mode - Watch the AI Learn!)")
        print("="*60)
        print(f"Target Episodes: {num_episodes}")
        print(f"Device: {agent.device}")
        print(f"Speed: {render_speed} cells/sec | FPS: {render_fps}")
        print("Note: Visual training is slower but you can watch!")
    else:
        print("Starting Training (Headless Mode - Maximum Speed)")
        print("="*60)
        print(f"Target Episodes: {num_episodes}")
        print(f"Device: {agent.device}")

    if use_hamilton_guidance and env.hamilton_planner:
        print(f"Hamilton Guidance: ENABLED")
        print(f"  Start: {hamilton_epsilon_start*100:.0f}% follow Hamilton")
        print(f"  End: {hamilton_epsilon_end*100:.0f}% follow Hamilton")
        print(f"  Strategy: Learn shortcuts while following optimal path")
    else:
        print("Hamilton Guidance: DISABLED (training from scratch)")
    print("="*60 + "\n")

    start_time = time.time()

    for episode in range(agent.n_games, num_episodes):
        state = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            # SAFETY MECHANISM: Get safe actions before choosing
            safe_actions = env._get_safe_actions()

            # Get Hamilton direction if guidance is enabled
            hamilton_dir = None
            followed_hamilton = False
            if use_hamilton_guidance and env.hamilton_planner:
                head_pos = env.snake_body[0]
                hamilton_dir = env.hamilton_planner.get_next_direction(head_pos, env.direction)
                # Track Hamilton usage
                if random.random() < hamilton_epsilon:
                    hamilton_follows += 1

            # Get action from agent (with Hamilton guidance and safety constraints)
            action = agent.get_action(state, training=True, safe_actions=safe_actions,
                                     hamilton_direction=hamilton_dir, hamilton_epsilon=hamilton_epsilon)

            # Check if agent followed Hamilton
            if hamilton_dir is not None:
                followed_hamilton = (action == hamilton_dir)

            # Take action in environment
            next_state, reward, done = env.step(action)

            # REWARD SHAPING: Bonus for successful shortcuts
            if use_hamilton_guidance and not followed_hamilton and not done:
                # Agent deviated from Hamilton and survived - small bonus
                reward += 0.5
            elif use_hamilton_guidance and not followed_hamilton and done:
                # Agent deviated from Hamilton and died - extra penalty
                reward -= 20

            # Render if visual mode is enabled
            if render:
                frames_per_move = max(1, int(render_fps / render_speed))
                for _ in range(frames_per_move):
                    env.render_frame()

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train on batch
            loss = agent.train_step()

            state = next_state
            score += reward
            steps += 1

        # Update metrics
        agent.n_games += 1
        scores.append(env.score)
        recent_scores.append(env.score)
        total_score += env.score
        mean_score = total_score / agent.n_games
        mean_scores.append(mean_score)

        if env.score > record:
            record = env.score

        # Decay exploration rate (ADAPTIVE - uses current score)
        agent.decay_epsilon(current_score=env.score)

        # Decay Hamilton guidance (curriculum learning)
        if use_hamilton_guidance:
            hamilton_epsilon = max(hamilton_epsilon_end, hamilton_epsilon * hamilton_epsilon_decay)

        # Anneal beta for prioritized replay
        agent.anneal_beta()

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Save model periodically
        if episode % save_freq == 0 and episode > 0:
            agent.save(model_path)

        # Check for best model (based on last 100 games average)
        # Only check and save every 50 episodes to reduce file I/O overhead
        if len(recent_scores) >= 100 and episode % 50 == 0:
            current_avg = np.mean(recent_scores)
            # Only save if significantly better (at least 0.5 improvement)
            if current_avg > best_avg_score + 0.5:
                best_avg_score = current_avg
                episodes_without_improvement = 0
                # Save best model
                agent.save(best_model_path)
                if not render:  # Don't spam in visual mode
                    print(f"  🌟 New best average score: {best_avg_score:.1f} (saved to {best_model_path})")
            elif current_avg < best_avg_score:
                episodes_without_improvement += 1

        # Early stopping check
        if early_stopping and len(recent_scores) >= 100:
            if episodes_without_improvement >= patience:
                print(f"\n⚠️  Early stopping triggered after {patience} episodes without improvement")
                print(f"Best average score: {best_avg_score:.1f}")
                break

        # Print progress
        if episode % 10 == 0 or episode == num_episodes - 1:
            elapsed = time.time() - start_time
            games_per_sec = (episode + 1 - agent.n_games + episode + 1) / elapsed if elapsed > 0 else 0
            recent_mean = np.mean(recent_scores) if recent_scores else 0

            if use_hamilton_guidance and env.hamilton_planner:
                print(f"Episode {episode:4d} | "
                      f"Score: {env.score:3d} | "
                      f"Record: {record:3d} | "
                      f"Avg(100): {recent_mean:5.1f} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Hamilton: {hamilton_epsilon:.3f} | "
                      f"Steps: {steps:4d}")
            else:
                print(f"Episode {episode:4d} | "
                      f"Score: {env.score:3d} | "
                      f"Record: {record:3d} | "
                      f"Avg(100): {recent_mean:5.1f} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Steps: {steps:4d}")

    # Final save
    agent.save(model_path)

    # Training summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total Episodes: {len(scores)}")
    print(f"Best Score: {record}")
    print(f"Final Avg Score (last 100): {np.mean(list(recent_scores)[-100:]):.1f}")
    if best_avg_score > 0:
        print(f"Best Avg Score (100-game window): {best_avg_score:.1f}")
    print(f"Total Time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Model saved to: {model_path}")
    if best_avg_score > 0:
        print(f"Best model saved to: {best_model_path}")
    print("="*60 + "\n")

    # Plot results
    plot_results(scores, mean_scores)


def plot_results(scores, mean_scores):
    """Plot training progress"""
    plt.figure(figsize=(12, 5))

    # Plot scores
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6, label='Score per Episode')
    plt.plot(mean_scores, label='Mean Score', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot moving average (last 100 games)
    plt.subplot(1, 2, 2)
    if len(scores) >= 100:
        moving_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
        plt.plot(moving_avg, label='Moving Avg (100 games)', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        plt.title('Moving Average Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Training plot saved to: training_progress.png")
    plt.show()


def interactive_menu():
    """
    Interactive CLI menu for training options
    """
    print("\n" + "="*60)
    print("🐍 SNAKE AI TRAINING - Enhanced Edition")
    print("="*60)
    print("\nChoose training mode:\n")
    print("1. Headless Training (RECOMMENDED)")
    print("   - No visualization")
    print("   - Maximum speed (~30-90 min for 5000 episodes)")
    print("   - Best for serious training")
    print("")
    print("2. Visual Training")
    print("   - Watch the AI learn in real-time")
    print("   - Much slower (~10-20x)")
    print("   - Great for understanding/debugging")
    print("")
    print("3. Quick Test (500 episodes, headless)")
    print("   - Fast test run (~5-10 minutes)")
    print("   - See if everything works")
    print("")
    print("4. Quick Visual Demo (100 episodes)")
    print("   - Short visual training session")
    print("   - Watch a few learning phases")
    print("="*60)

    while True:
        choice = input("\nEnter choice (1-4) or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            print("Exiting...")
            sys.exit(0)

        if choice in ['1', '2', '3', '4']:
            return choice

        print("Invalid choice. Please enter 1, 2, 3, 4, or 'q'")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Snake AI with Enhanced Deep Reinforcement Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                    # Interactive menu
  python train.py --headless         # Fast headless training (5000 episodes)
  python train.py --visual           # Visual training (500 episodes)
  python train.py --episodes 1000    # Custom episode count
  python train.py --quick            # Quick test (500 episodes)
        """
    )

    parser.add_argument('--headless', action='store_true',
                        help='Headless training mode (no visualization, fast)')
    parser.add_argument('--visual', action='store_true',
                        help='Visual training mode (watch AI learn, slow)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (500 episodes, headless)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes (overrides mode defaults)')
    parser.add_argument('--speed', type=int, default=20,
                        help='Speed in cells/sec for visual mode (default: 20)')
    parser.add_argument('--model', type=str, default='snake_model.pth',
                        help='Model save path (default: snake_model.pth)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine mode
    if args.headless or args.visual or args.quick:
        # Command-line mode
        if args.headless:
            mode = '1'
        elif args.visual:
            mode = '2'
        elif args.quick:
            mode = '3'
    else:
        # Interactive mode
        mode = interactive_menu()

    # Configure based on mode
    if mode == '1':  # Headless
        print("\n🚀 Starting HEADLESS training (maximum speed)...")
        train_config = {
            'num_episodes': args.episodes or 5000,
            'target_update_freq': 10,
            'save_freq': 100,
            'model_path': args.model,
            'render': False
        }

    elif mode == '2':  # Visual
        print("\n👀 Starting VISUAL training (watch it learn)...")
        train_config = {
            'num_episodes': args.episodes or 500,
            'target_update_freq': 10,
            'save_freq': 25,
            'model_path': args.model,
            'render': True,
            'render_fps': 60,
            'render_speed': args.speed,
            'render_hamilton': True  # Show Hamilton path in visual mode
        }

    elif mode == '3':  # Quick test
        print("\n⚡ Starting QUICK TEST (500 episodes, headless)...")
        train_config = {
            'num_episodes': args.episodes or 500,
            'target_update_freq': 10,
            'save_freq': 50,
            'model_path': args.model,
            'render': False
        }

    elif mode == '4':  # Quick visual demo
        print("\n🎬 Starting QUICK VISUAL DEMO (100 episodes)...")
        train_config = {
            'num_episodes': args.episodes or 100,
            'target_update_freq': 10,
            'save_freq': 25,
            'model_path': args.model,
            'render': True,
            'render_fps': 60,
            'render_speed': args.speed,
            'render_hamilton': True  # Show Hamilton path in visual demo
        }

    # Show configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION:")
    print("="*60)
    for key, value in train_config.items():
        print(f"  {key:20s}: {value}")
    print("="*60)

    # Confirm before starting
    if not args.headless and not args.visual and not args.quick:
        confirm = input("\nStart training? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Training cancelled.")
            sys.exit(0)

    print("\n🎮 Training starting...\n")

    # Start training
    train(**train_config)
