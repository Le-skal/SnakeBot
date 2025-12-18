"""
Deep Q-Network (DQN) Agent for Snake
Lightweight implementation optimized for modest hardware
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """
    Optimized CNN for Q-value prediction with spatial awareness and lookahead
    Now supports DUELING architecture for better value estimation

    Input structure (124 features):
    - Immediate danger (3): 1 cell ahead
    - Lookahead danger (3): 2 cells ahead
    - Current direction (4)
    - Food direction (4)
    - Grid (100) - reshaped to 10x10, with head/body/tail differentiated
    - Snake length (1)
    - Accessible space (1)
    - Can reach tail (1)
    - Path to food (1)
    - Space after eating (1)
    - Hamilton direction (4)
    - Should follow Hamilton (1)
    """
    def __init__(self, input_size, hidden_size, output_size, grid_width=10, grid_height=10, dueling=True):
        super(DQN, self).__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_size = grid_width * grid_height
        self.dueling = dueling
        self.output_size = output_size

        # Efficient CNN layers with pooling
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 10x10 -> 10x10
        self.pool1 = nn.MaxPool2d(2, 2)  # 10x10 -> 5x5
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 5x5 -> 5x5

        # Global average pooling reduces 32 channels of 5x5 to just 32 features
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 5x5 -> 1x1 per channel

        # Number of non-grid features: 3 + 3 + 4 + 4 + 1 + 1 + 1 + 1 + 1 + 4 + 1 = 24
        non_grid_features = 24

        # Compact fully connected layers
        # CNN gives 32 features + 24 non-grid = 56 total
        self.fc1 = nn.Linear(32 + non_grid_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)

        if dueling:
            # DUELING architecture: separate Value and Advantage streams
            # Value stream: estimates how good it is to be in this state
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)  # Single value for the state
            )

            # Advantage stream: estimates how much better each action is
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, output_size)  # Advantage for each action
            )
        else:
            # Standard DQN: direct Q-value output
            self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Split input into grid and non-grid features
        # Immediate danger (3) + Lookahead danger (3) + Direction (4) + Food (4) = 14
        non_grid_before = x[:, :14]  # All features before grid
        grid_start = 14
        grid_end = 14 + self.grid_size
        grid = x[:, grid_start:grid_end]  # Grid (grid_width * grid_height elements)
        non_grid_after = x[:, grid_end:]  # Snake length + new features (10 features)

        # Reshape grid for CNN: (batch, 1, grid_height, grid_width)
        grid = grid.view(batch_size, 1, self.grid_height, self.grid_width)

        # Process grid through CNN layers with pooling
        x_grid = torch.relu(self.conv1(grid))  # -> 16 x 10 x 10
        x_grid = self.pool1(x_grid)  # -> 16 x 5 x 5
        x_grid = torch.relu(self.conv2(x_grid))  # -> 32 x 5 x 5

        # Global average pooling: 32 x 5 x 5 -> 32 x 1 x 1
        x_grid = self.global_pool(x_grid)
        x_grid = x_grid.view(batch_size, -1)  # -> 32 features

        # Combine CNN features with non-grid features
        x_combined = torch.cat([x_grid, non_grid_before, non_grid_after], dim=1)

        # Process through shared fully connected layers
        x_combined = torch.relu(self.fc1(x_combined))
        x_combined = torch.relu(self.fc2(x_combined))

        if self.dueling:
            # DUELING: compute Value and Advantage separately
            value = self.value_stream(x_combined)  # (batch, 1)
            advantage = self.advantage_stream(x_combined)  # (batch, num_actions)

            # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            # Subtracting mean makes the advantage centered around zero
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            # Standard DQN
            return self.fc3(x_combined)


class ReplayMemory:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay Buffer
    Samples important experiences more frequently
    """
    def __init__(self, capacity, alpha=0.6):
        """
        Args:
            capacity: Maximum size of buffer
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Save a transition with maximum priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions based on priorities

        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling weight (0 = no correction, 1 = full correction)

        Returns:
            batch: List of transitions
            indices: Indices of sampled transitions (for updating priorities)
            weights: Importance sampling weights
        """
        if len(self.memory) == 0:
            return [], [], []

        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.memory)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities, replace=False)

        # Calculate importance sampling weights
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        # Get samples
        batch = [self.memory[idx] for idx in indices]

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(
        self,
        state_size=11,
        action_size=4,
        hidden_size=256,
        lr=0.001,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=100_000,
        batch_size=1000,
        use_prioritized_replay=True,
        adaptive_epsilon=True,
        grid_width=10,
        grid_height=10
    ):
        """
        Initialize DQN Agent (ENHANCED)

        Args:
            state_size: Size of state vector (119 for enhanced snake)
            action_size: Number of possible actions (4: up, down, left, right)
            hidden_size: Size of hidden layers in neural network
            lr: Learning rate
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of replay memory
            batch_size: Batch size for training
            use_prioritized_replay: Use prioritized experience replay
            adaptive_epsilon: Use adaptive epsilon decay based on performance
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.use_prioritized_replay = use_prioritized_replay
        self.adaptive_epsilon = adaptive_epsilon
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Policy network (the one we train) - with DUELING architecture
        self.policy_net = DQN(state_size, hidden_size, action_size,
                             grid_width=grid_width, grid_height=grid_height, dueling=True).to(self.device)

        # Target network (for stable learning) - with DUELING architecture
        self.target_net = DQN(state_size, hidden_size, action_size,
                             grid_width=grid_width, grid_height=grid_height, dueling=True).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        print("Using Dueling Double DQN architecture")

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Choose memory type
        if use_prioritized_replay:
            self.memory = PrioritizedReplayMemory(memory_size, alpha=0.6)
            print("Using Prioritized Experience Replay")
        else:
            self.memory = ReplayMemory(memory_size)
            print("Using Standard Experience Replay")

        self.n_games = 0
        self.best_score = 0
        self.beta = 0.4  # For prioritized replay importance sampling

    def get_action(self, state, training=True, safe_actions=None, hamilton_direction=None, hamilton_epsilon=0.8):
        """
        Choose action using Hamilton-guided epsilon-greedy policy

        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, always use best action
            safe_actions: List of safe actions (for emergency fallback)
            hamilton_direction: Optimal Hamilton cycle direction (if available)
            hamilton_epsilon: Probability of following Hamilton path (default 0.8 = 80%)

        Returns:
            action: Index of chosen action
        """
        # HAMILTON-GUIDED LEARNING: Follow Hamilton path most of the time, explore occasionally
        if training and hamilton_direction is not None and random.random() < hamilton_epsilon:
            # Follow Hamilton cycle (teacher policy)
            # This provides a strong baseline and prevents bad exploration
            if hamilton_direction in (safe_actions if safe_actions else range(self.action_size)):
                return hamilton_direction
            # If Hamilton direction is unsafe (rare), fall through to Q-network

        # Standard epsilon-greedy for exploration/exploitation
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            # But still prefer safe actions if available
            if safe_actions and len(safe_actions) > 0:
                return random.choice(safe_actions)
            else:
                return random.randint(0, self.action_size - 1)
        else:
            # Best action according to Q-network (exploitation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)

            # SAFETY MECHANISM: Only choose from safe actions
            if safe_actions and len(safe_actions) > 0:
                # Mask out unsafe actions
                q_values_np = q_values.cpu().numpy()[0]
                masked_q = {action: q_values_np[action] for action in safe_actions}
                return max(masked_q, key=masked_q.get)
            else:
                # No safe actions available (shouldn't happen, but fallback to best Q)
                return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step (if enough samples in memory) - ENHANCED"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from memory
        if self.use_prioritized_replay:
            # Prioritized replay: sample based on TD error
            batch, indices, weights = self.memory.sample(self.batch_size, beta=self.beta)
            if not batch:
                return None

            states, actions, rewards, next_states, dones = zip(*batch)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            # Standard replay: uniform sampling
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Next Q values from target network (Double DQN)
        with torch.no_grad():
            # Use policy network to SELECT action
            next_actions = self.policy_net(next_states).argmax(1)
            # Use target network to EVALUATE action (Double DQN trick)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute TD errors for prioritized replay
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()

        # Compute weighted loss
        loss = (weights * (current_q_values - target_q_values) ** 2).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Update priorities in prioritized replay
        if self.use_prioritized_replay and indices is not None:
            # Add small constant to avoid zero priority
            priorities = td_errors + 1e-6
            self.memory.update_priorities(indices, priorities)

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self, current_score=0):
        """
        Reduce exploration rate - ADAPTIVE version
        Decays faster when performance improves, slower when stuck

        Args:
            current_score: Current game score for adaptive decay
        """
        if self.adaptive_epsilon:
            # Adaptive epsilon: decay faster when doing well
            if current_score > self.best_score:
                self.best_score = current_score
                # Breakthrough! Decay faster to exploit what we learned
                decay_multiplier = 0.98
            elif current_score >= self.best_score * 0.8:
                # Doing well, normal decay
                decay_multiplier = self.epsilon_decay
            else:
                # Struggling, decay slower (keep exploring)
                decay_multiplier = 0.999

            self.epsilon = max(self.epsilon_end, self.epsilon * decay_multiplier)
        else:
            # Standard epsilon decay
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def anneal_beta(self):
        """Anneal beta for prioritized replay (gradually increase importance sampling correction)"""
        if self.use_prioritized_replay:
            self.beta = min(1.0, self.beta + 0.001)

    def save(self, filename):
        """Save model to file"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'n_games': self.n_games
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load model from file"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.n_games = checkpoint['n_games']
        print(f"Model loaded from {filename}")
