# 🐍 Snake AI Bot

An advanced Snake game implementation featuring multiple AI approaches, including a perfect Hamilton cycle algorithm and a Deep Q-Learning neural network.

## 🌟 Features

- **🎮 Manual Play Mode** - Control the snake yourself with smooth animations
- **🤖 Hamilton Cycle Algorithm** - Watch a perfect AI that never dies and completes every grid
- **🧠 Deep Q-Learning AI** - Train a neural network to play Snake using reinforcement learning
- **📊 Advanced State Representation** - 124-dimensional state space with spatial awareness
- **🎯 Dueling DQN Architecture** - Separate value and advantage streams for better learning
- **⚡ Prioritized Experience Replay** - Learn from important experiences more efficiently
- **📈 Real-time Visualization** - Watch training progress with smooth pygame rendering
- **🔄 Resizable Windows** - Drag to resize or press F11 for fullscreen
- **🗺️ Hamilton Path Overlay** - Toggle with 'H' to see the optimal path visualization

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Game Modes](#game-modes)
- [Architecture](#architecture)
- [Training Details](#training-details)
- [Project Structure](#project-structure)
- [Controls](#controls)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Performance](#performance)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.24.0` - Numerical computations
- `pygame>=2.5.0` - Game rendering and visualization
- `matplotlib>=3.7.0` - Plotting training metrics

## 🎯 Quick Start

Run the main program:

```bash
python main.py
```

You'll see an interactive menu with 5 options:

```
🐍  SNAKE AI GAME  🐍

1. 🎮 Play Manually (You control the snake)
2. 🤖 Watch Hamilton Cycle Demo (Perfect AI play)
3. 🧠 Train Neural Network (Deep Q-Learning)
4. 👀 Watch Trained Model Play
5. 🚪 Exit
```

## 🎮 Game Modes

### 1. Manual Play 🎮

Control the snake yourself and try to beat your high score!

**Controls:**
- `Arrow Keys` or `WASD` - Move the snake
- `R` or `Space` - Restart game
- `H` - Toggle Hamilton path overlay
- `F11` - Toggle fullscreen
- `ESC` - Exit

**Features:**
- Smooth interpolated movement
- Resizable window
- Real-time scoring
- Hamilton path visualization (optional)

### 2. Hamilton Cycle Demo 🤖

Watch an AI that follows a predetermined Hamiltonian cycle - a path that visits every cell exactly once.

**How it works:**
1. Uses Prim's algorithm to generate a minimum spanning tree on a half-resolution grid
2. Navigates around the tree edges to create a Hamiltonian cycle
3. Snake follows this path indefinitely, eventually filling the entire grid
4. Guaranteed to never die (unless interrupted)

**Features:**
- Works on ANY grid size (even, odd, rectangular)
- Generates unique random patterns each time
- Optional path visualization overlay
- Configurable speed

**Use Cases:**
- Baseline comparison for AI training
- Demonstrate solvability of Snake
- Generate training data for neural networks

### 3. Train Neural Network 🧠

Train a Deep Q-Learning agent from scratch using reinforcement learning.

**Training Features:**
- **Dueling Double DQN Architecture**: Separates state value from action advantages
- **Prioritized Experience Replay**: Learns from important experiences more frequently
- **Adaptive Epsilon Decay**: Adjusts exploration based on performance
- **Multi-step Danger Detection**: Looks ahead 1 and 2 cells
- **Spatial Awareness**: Full grid representation with CNN processing
- **Hamilton Guidance**: Optional teacher policy for faster learning
- **Advanced Reward Shaping**: Encourages safe, strategic play

**State Representation (124 features):**
- Immediate danger (3): collision risk 1 cell ahead
- Lookahead danger (3): collision risk 2 cells ahead
- Current direction (4): one-hot encoded
- Food direction (4): relative position to head
- Full grid (100): spatial awareness (10×10 grid)
  - Head = 1.5
  - Body = 1.0
  - Tail = 0.5
  - Food = 2.0
- Snake length (1): normalized
- Accessible space (1): reachable empty cells
- Can reach tail (1): connectivity check
- Path to food (1): BFS distance
- Space after eating (1): lookahead space
- Hamilton direction (4): optimal path guidance
- Should follow Hamilton (1): binary flag

**Training Options:**
- Headless mode (fast) or visual mode (watchable)
- Configurable grid size
- Adjustable training episodes
- Auto-saves best models

### 4. Watch Trained Model 👀

Load a saved model and watch it play multiple games.

**Features:**
- Visualize learned strategies
- Compare performance across different models
- Optional Hamilton path overlay
- Configurable playback speed
- Statistics tracking

## 🏗️ Architecture

### Deep Q-Network (DQN) Structure

```
Input (124 features)
    ↓
Split into Grid (100) and Non-Grid (24)
    ↓                         ↓
CNN Branch               Feature Branch
    ↓                         ↓
Conv2d(1→16, 3×3)           Pass through
    ↓
MaxPool(2×2)
    ↓
Conv2d(16→32, 3×3)
    ↓
AdaptiveAvgPool
    ↓
    └─────────┬──────────────┘
              ↓
        Concatenate (56)
              ↓
         FC(56→256)
              ↓
         FC(256→128)
              ↓
      ┌───────┴───────┐
      ↓               ↓
Value Stream    Advantage Stream
  FC(128→32)       FC(128→32)
      ↓               ↓
   FC(32→1)        FC(32→4)
      ↓               ↓
      └───────┬───────┘
              ↓
    Q(s,a) = V(s) + (A(s,a) - mean(A))
              ↓
         Q-Values (4)
```

### Hamilton Cycle Algorithm

1. **Grid Preparation**: Create nodes at odd coordinates (half-resolution)
2. **Graph Construction**: Build weighted edges between adjacent nodes
3. **MST Generation**: Apply Prim's algorithm to create spanning tree
4. **Cycle Navigation**: Use wall-following to traverse around tree edges
5. **Path Mapping**: Convert to full-resolution Hamiltonian cycle

## 📚 Training Details

### Reward Structure

- **Eating food**: +100 (encourages growth)
- **Following Hamilton (when large)**: +20 (bonus for safety)
- **Space preservation**: Variable (encourages open space)
- **Moving toward food**: +0.1 (small hint when safe)
- **Reducing accessible space**: -0.5 per cell (discourages trapping)
- **Low space ratio**: -1.0 (warns of cramped conditions)
- **Wall collision**: -100 (severe penalty)
- **Self collision**: -100 (severe penalty)

### Hyperparameters

```python
learning_rate = 0.001          # Adam optimizer
gamma = 0.9                    # Discount factor
epsilon_start = 1.0            # Initial exploration
epsilon_end = 0.01             # Minimum exploration
epsilon_decay = 0.995          # Decay rate
memory_size = 100,000          # Replay buffer
batch_size = 1000              # Training batch
hidden_size = 256              # Network width
target_update = 10             # Update frequency
```

### Training Tips

1. **Start with small grids** (10×10) - easier to learn
2. **Use headless mode** for faster training
3. **Monitor episode scores** - should gradually increase
4. **Enable Hamilton guidance** - speeds up early learning
5. **Train for 1000+ episodes** - patience is key
6. **Save checkpoints** - resume training later

## 📁 Project Structure

```
Snake/
├── main.py                    # Interactive menu and launcher
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── algorithms/
│   ├── __init__.py
│   ├── hamilton_cycle.py     # Hamiltonian cycle generation
│   └── inspiration.py        # Original cycle visualization
│
├── game/
│   ├── __init__.py
│   ├── environment.py        # RL environment (SnakeEnv)
│   └── manual_play.py        # Human playable game
│
├── training/
│   ├── __init__.py
│   ├── snake_ai.py           # DQN agent implementation
│   ├── train.py              # Training loop
│   └── watch.py              # Model evaluation
│
└── demos/
    ├── __init__.py
    └── hamilton_demo.py      # Hamilton cycle demo
```

## ⌨️ Controls

### During Gameplay

| Key | Action |
|-----|--------|
| `↑` / `W` | Move up |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `→` / `D` | Move right |
| `R` / `Space` | Restart game |
| `H` | Toggle Hamilton path overlay |
| `F11` | Toggle fullscreen |
| `ESC` | Exit |

### Window Management

- **Drag window edges** - Resize (maintains aspect ratio)
- **F11** - Fullscreen mode
- **ESC** - Exit fullscreen or quit

## ⚙️ Configuration

### Grid Size

Customize grid dimensions when prompted:
- Default: 10×10
- Supports rectangular grids (e.g., 20×15)
- Larger grids = harder to learn (more state space)

### Speed

Adjust movement speed (cells per second):
- Manual play: 6 cells/sec (default)
- Hamilton demo: Configurable (1-20)
- Training: Auto-optimized

### Training Episodes

- Quick test: 100 episodes
- Basic training: 1,000 episodes
- Advanced training: 10,000+ episodes

## 🔧 Technical Details

### State Space Complexity

- **Total state space**: ~10^30 possible states
- **Compressed representation**: 124 real-valued features
- **CNN processing**: Extracts spatial patterns from 10×10 grid
- **Feature engineering**: Hand-crafted danger/food signals

### Action Space

4 discrete actions: UP, DOWN, LEFT, RIGHT
- Invalid moves (180° turns) automatically prevented
- Safe action masking during inference

### Exploration Strategy

1. **Hamilton-guided**: Follow optimal path 80% of the time (early training)
2. **Epsilon-greedy**: Random exploration with decay
3. **Adaptive decay**: Faster when improving, slower when stuck
4. **Safe action filtering**: Prefer non-collision moves

### Learning Algorithm

**Dueling Double DQN** with:
- **Double DQN**: Reduces overestimation bias
  - Policy network selects action
  - Target network evaluates action
- **Dueling Architecture**: Separates value and advantage
  - V(s): How good is this state?
  - A(s,a): How much better is each action?
  - Q(s,a) = V(s) + (A(s,a) - mean(A))
- **Prioritized Replay**: Samples high-TD-error experiences more
- **Target Network**: Updated every 10 episodes for stability
- **Gradient Clipping**: Prevents exploding gradients

## 📊 Performance

### Hamilton Cycle

- **Success Rate**: 100% (never dies)
- **Grid Completion**: Fills entire grid every time
- **Speed**: Depends on grid size (10×10 takes ~100 moves)
- **Memory**: O(grid_size) - stores cycle mapping

### Deep Q-Learning

Typical performance after 1000 episodes on 10×10 grid:
- **Average Score**: 15-30 (varies by training run)
- **Max Score**: 40-98 (best agents approach perfection)
- **Training Time**: 1-3 hours (GPU) or 5-10 hours (CPU)
- **Model Size**: ~2 MB

### Hardware Requirements

**Minimum:**
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- GPU: Optional (CPU training works)

**Recommended:**
- CPU: Quad-core 3.0 GHz+
- RAM: 8 GB
- GPU: CUDA-compatible (10× faster training)

## 🧪 Advanced Usage

### Custom Grid Sizes

```python
from game.environment import SnakeEnv

env = SnakeEnv(
    render=True,
    grid_width=20,
    grid_height=15,
    speed_cells=8
)
```

### Load Pretrained Model

```python
from training.snake_ai import DQNAgent

agent = DQNAgent(state_size=124, action_size=4)
agent.load('snake_model.pth')
```

### Hamilton Path Only

```python
from algorithms.hamilton_cycle import HamiltonianSnakePlanner

planner = HamiltonianSnakePlanner(grid_width=10, grid_height=10)
next_dir = planner.get_next_direction(head_pos=[5, 5])
```

### Visualize Cycle

```python
from algorithms.hamilton_cycle import visualize_cycle

visualize_cycle(grid_width=10, grid_height=10)
```

## 🐛 Troubleshooting

### "No module named torch"
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### "pygame not found"
```bash
pip install pygame --upgrade
```

### Training is slow
- Use headless mode (disable rendering)
- Reduce grid size
- Use GPU if available
- Decrease batch size

### Model not improving
- Train longer (1000+ episodes minimum)
- Adjust reward structure
- Enable Hamilton guidance
- Check epsilon decay rate

### Window too small/large
- Press F11 for fullscreen
- Drag window edges to resize
- Adjust grid size in configuration

## 📖 Learning Resources

### Reinforcement Learning
- [Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep Q-Learning Paper (2015)](https://www.nature.com/articles/nature14236)
- [Dueling DQN Paper (2016)](https://arxiv.org/abs/1511.06581)

### Hamiltonian Cycles
- [Hamiltonian Path Problem](https://en.wikipedia.org/wiki/Hamiltonian_path_problem)
- [Prim's Algorithm](https://en.wikipedia.org/wiki/Prim%27s_algorithm)

### Snake AI Strategies
- [Perfect Snake AI Analysis](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Multi-snake competitive mode
- Curriculum learning schedules
- Different neural architectures (transformers?)
- Online learning / continuous training
- Tournament mode with leaderboards

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Hamilton cycle implementation inspired by classic Nokia Snake algorithms
- Deep Q-Learning based on DeepMind's DQN papers
- Smooth rendering adapted from pygame community examples

## 📧 Contact

For questions, issues, or suggestions, please open an issue on the project repository.

---

**Made with 🐍 and ❤️**

*Happy Snake Training!* 🎮🤖