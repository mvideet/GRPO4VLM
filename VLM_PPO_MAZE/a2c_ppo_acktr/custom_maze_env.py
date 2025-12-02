"""
Custom Maze Environment using the bitmask-based maze generator.
Replaces gym-maze with our own maze generation system.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

# Bitmask constants for walls
N, S, E, W = 1, 2, 4, 8
DX = {E: 1, W: -1, N: 0, S: 0}
DY = {E: 0, W: 0, N: -1, S: 1}
OPPOSITE = {E: W, W: E, N: S, S: N}


def generate_maze(width=5, height=5, rng=None):
    """
    Generate a random maze using recursive backtracking.
    
    Returns:
        grid: HxW 2D list of bitmasks; each cell's bits tell you which walls remain.
    """
    if rng is None:
        rng = random

    # Start with all walls present in each cell
    grid = [[N | S | E | W for _ in range(width)] for _ in range(height)]
    visited = [[False] * width for _ in range(height)]

    def carve(x, y):
        visited[y][x] = True
        directions = [N, S, E, W]
        rng.shuffle(directions)

        for direction in directions:
            nx = x + DX[direction]
            ny = y + DY[direction]
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                # Knock down wall between current cell and neighbor
                grid[y][x] &= ~direction
                grid[ny][nx] &= ~OPPOSITE[direction]
                carve(nx, ny)

    # Start carving from top-left
    carve(0, 0)
    return grid


class CustomMazeEnv(gym.Env):
    """
    Custom maze environment using bitmask-based maze generation.
    Agent starts at (0,0) and goal is at (height-1, width-1).
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, width=5, height=5, max_steps=200, seed=None):
        super().__init__()
        self.width = width
        self.height = height
        self.maze_size = (height, width)
        self.max_steps = max_steps
        self.step_count = 0
        
        # Action space: 0=N, 1=S, 2=E, 3=W
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent position [row, col]
        self.observation_space = spaces.Box(
            low=0, high=max(width, height), shape=(2,), dtype=np.float32
        )
        
        # Internal state
        self.maze_grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.rng = random.Random(seed) if seed is not None else random
        
        # Generate initial maze
        self._generate_maze()
    
    def _generate_maze(self):
        """Generate a new random maze."""
        self.maze_grid = generate_maze(self.width, self.height, rng=self.rng)
        self.agent_pos = np.array([0, 0], dtype=np.float32)  # Start at top-left
        self.goal_pos = np.array([self.height - 1, self.width - 1], dtype=np.float32)  # Goal at bottom-right
        self.step_count = 0
    
    def _can_move(self, pos, direction):
        """
        Check if agent can move in given direction from position.
        Returns (can_move, new_pos)
        """
        x, y = int(pos[1]), int(pos[0])  # pos is [row, col]
        
        # Check bounds
        if direction == N and y == 0:
            return False, pos
        if direction == S and y == self.height - 1:
            return False, pos
        if direction == W and x == 0:
            return False, pos
        if direction == E and x == self.width - 1:
            return False, pos
        
        # Check wall
        cell = self.maze_grid[y][x]
        if cell & direction:  # Wall in this direction
            return False, pos
        
        # Can move
        new_x = x + DX[direction]
        new_y = y + DY[direction]
        new_pos = np.array([new_y, new_x], dtype=np.float32)
        return True, new_pos
    
    def reset(self, seed=None, options=None):
        """Reset environment and generate new maze."""
        if seed is not None:
            self.rng = random.Random(seed)
            np.random.seed(seed)
        
        self._generate_maze()
        
        info = {
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'maze_grid': self.maze_grid
        }
        
        return self.agent_pos.copy(), info
    
    def step(self, action):
        """
        Execute action.
        Action: 0=N, 1=S, 2=E, 3=W
        """
        direction_map = {0: N, 1: S, 2: E, 3: W}
        direction = direction_map.get(action, N)
        
        # Try to move
        can_move, new_pos = self._can_move(self.agent_pos, direction)
        
        if can_move:
            self.agent_pos = new_pos
        
        self.step_count += 1
        
        # Check if reached goal
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        terminated = distance < 0.5
        
        # Check if max steps reached
        truncated = self.step_count >= self.max_steps
        
        # Reward (will be computed by DenseRewardWrapper)
        reward = 0.0
        
        info = {
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'maze_grid': self.maze_grid,
            'distance': distance,
            'reached_goal': terminated
        }
        
        return self.agent_pos.copy(), reward, terminated, truncated, info
    
    
    def get_maze_grid(self):
        """Get the bitmask maze grid."""
        return self.maze_grid

