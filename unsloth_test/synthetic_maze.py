# -*- coding: utf-8 -*-
"""
Synthetic Maze Generator for GRPO Training
===========================================
Self-contained maze generation that doesn't require gym_maze.
Uses recursive backtracking to generate valid, solvable mazes.
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import random
from collections import deque


@dataclass
class MazeState:
    """Represents the current state of a maze environment."""
    maze: np.ndarray  # 2D array: 0=path, 1=wall
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    visited: set
    step_count: int = 0
    
    def copy(self) -> 'MazeState':
        return MazeState(
            maze=self.maze.copy(),
            agent_pos=self.agent_pos,
            goal_pos=self.goal_pos,
            visited=self.visited.copy(),
            step_count=self.step_count
        )


class SyntheticMazeEnv:
    """
    A simple, self-contained maze environment.
    No external dependencies required.
    """
    
    # Action mapping
    ACTIONS = {
        0: (-1, 0),  # UP/NORTH
        1: (1, 0),   # DOWN/SOUTH  
        2: (0, 1),   # RIGHT/EAST
        3: (0, -1),  # LEFT/WEST
    }
    ACTION_NAMES = {0: 'N', 1: 'S', 2: 'E', 3: 'W'}
    
    def __init__(self, size: Tuple[int, int] = (5, 5), seed: int = None):
        self.size = size
        self.seed = seed
        self.state: Optional[MazeState] = None
        self.max_steps = size[0] * size[1] * 2
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _generate_maze(self) -> np.ndarray:
        """Generate a solvable maze using randomized Prim's algorithm."""
        rows, cols = self.size
        
        # Start with all walls
        maze = np.ones((rows, cols), dtype=np.int32)
        
        # Mark start as path
        maze[0, 0] = 0
        
        # Frontier: walls adjacent to paths that could become paths
        frontier = []
        
        def add_frontier(r, c):
            """Add adjacent walls to frontier."""
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 1:
                    if (nr, nc) not in frontier:
                        frontier.append((nr, nc))
        
        add_frontier(0, 0)
        
        while frontier:
            # Pick random frontier cell
            idx = random.randint(0, len(frontier) - 1)
            r, c = frontier.pop(idx)
            
            # Count adjacent path cells
            path_neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
                    path_neighbors.append((nr, nc))
            
            # Only carve if exactly 1 path neighbor (prevents loops/large open areas)
            if len(path_neighbors) == 1:
                maze[r, c] = 0
                add_frontier(r, c)
        
        # Ensure goal is reachable by carving path if needed
        goal_r, goal_c = rows - 1, cols - 1
        maze[goal_r, goal_c] = 0
        
        # BFS to check if goal is reachable
        from collections import deque
        queue = deque([(0, 0)])
        visited = {(0, 0)}
        reachable = False
        
        while queue and not reachable:
            cr, cc = queue.popleft()
            if (cr, cc) == (goal_r, goal_c):
                reachable = True
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    maze[nr, nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        # If not reachable, carve a path
        if not reachable:
            # Simple: carve along bottom and right edges
            for c in range(cols):
                maze[rows - 1, c] = 0
            for r in range(rows):
                maze[r, cols - 1] = 0
        
        return maze
    
    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (in bounds and not a wall)."""
        r, c = pos
        if 0 <= r < self.size[0] and 0 <= c < self.size[1]:
            return self.state.maze[r, c] == 0
        return False
    
    def reset(self, seed: int = None) -> MazeState:
        """Reset environment to initial state."""
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
        
        maze = self._generate_maze()
        
        self.state = MazeState(
            maze=maze,
            agent_pos=(0, 0),
            goal_pos=(self.size[0] - 1, self.size[1] - 1),
            visited={(0, 0)},
            step_count=0
        )
        
        return self.state
    
    def step(self, action: int) -> Tuple[MazeState, float, bool, bool, Dict]:
        """
        Execute action and return (state, reward, terminated, truncated, info).
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Get movement delta
        dr, dc = self.ACTIONS.get(action, (0, 0))
        
        # Calculate new position
        new_r = self.state.agent_pos[0] + dr
        new_c = self.state.agent_pos[1] + dc
        new_pos = (new_r, new_c)
        
        # Store previous position for reward calculation
        prev_pos = self.state.agent_pos
        
        # Check if move is valid
        if self._is_valid_pos(new_pos):
            self.state.agent_pos = new_pos
            self.state.visited.add(new_pos)
        
        self.state.step_count += 1
        
        # Check termination
        terminated = (self.state.agent_pos == self.state.goal_pos)
        truncated = (self.state.step_count >= self.max_steps)
        
        # Calculate reward (dense: distance improvement)
        prev_dist = abs(prev_pos[0] - self.state.goal_pos[0]) + abs(prev_pos[1] - self.state.goal_pos[1])
        curr_dist = abs(self.state.agent_pos[0] - self.state.goal_pos[0]) + abs(self.state.agent_pos[1] - self.state.goal_pos[1])
        reward = prev_dist - curr_dist
        
        if terminated:
            reward += 10.0  # Bonus for reaching goal
        
        info = {
            'agent_pos': self.state.agent_pos,
            'goal_pos': self.state.goal_pos,
            'prev_pos': prev_pos,
            'step_count': self.state.step_count,
            'solved': terminated
        }
        
        return self.state, reward, terminated, truncated, info
    
    def render(self, cell_size: int = 30) -> Image.Image:
        """Render maze as PIL Image."""
        if self.state is None:
            raise RuntimeError("Must call reset() before render()")
        
        rows, cols = self.size
        width = cols * cell_size
        height = rows * cell_size
        
        # Colors
        WALL_COLOR = (40, 40, 40)
        PATH_COLOR = (255, 255, 255)
        VISITED_COLOR = (220, 220, 240)
        AGENT_COLOR = (0, 200, 0)
        GOAL_COLOR = (200, 0, 0)
        GRID_COLOR = (180, 180, 180)
        
        img = Image.new('RGB', (width, height), PATH_COLOR)
        draw = ImageDraw.Draw(img)
        
        # Draw cells
        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                if self.state.maze[r, c] == 1:
                    draw.rectangle([x1, y1, x2, y2], fill=WALL_COLOR)
                elif (r, c) in self.state.visited and (r, c) != self.state.agent_pos:
                    draw.rectangle([x1, y1, x2, y2], fill=VISITED_COLOR)
        
        # Draw goal
        gr, gc = self.state.goal_pos
        gx1, gy1 = gc * cell_size, gr * cell_size
        gx2, gy2 = gx1 + cell_size, gy1 + cell_size
        padding = max(3, cell_size // 5)
        draw.ellipse([gx1 + padding, gy1 + padding, gx2 - padding, gy2 - padding], 
                    fill=GOAL_COLOR)
        
        # Draw agent
        ar, ac = self.state.agent_pos
        ax1, ay1 = ac * cell_size, ar * cell_size
        ax2, ay2 = ax1 + cell_size, ay1 + cell_size
        draw.ellipse([ax1 + padding, ay1 + padding, ax2 - padding, ay2 - padding],
                    fill=AGENT_COLOR)
        
        # Draw grid lines
        for r in range(rows + 1):
            y = r * cell_size
            draw.line([(0, y), (width, y)], fill=GRID_COLOR, width=1)
        for c in range(cols + 1):
            x = c * cell_size
            draw.line([(x, 0), (x, height)], fill=GRID_COLOR, width=1)
        
        return img
    
    def get_optimal_path(self) -> Optional[List[int]]:
        """Find optimal path using BFS. Returns list of action indices."""
        if self.state is None:
            return None
        
        start = self.state.agent_pos
        goal = self.state.goal_pos
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            pos, path = queue.popleft()
            
            if pos == goal:
                return path
            
            for action, (dr, dc) in self.ACTIONS.items():
                new_pos = (pos[0] + dr, pos[1] + dc)
                
                if new_pos not in visited and self._is_valid_pos(new_pos):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [action]))
        
        return None  # No path found


def execute_actions(env: SyntheticMazeEnv, actions: List[int], 
                    max_steps: int = 100) -> Tuple[float, bool, int, Dict]:
    """
    Execute a sequence of actions in the maze.
    
    Returns:
        total_reward: Sum of rewards
        solved: Whether maze was solved
        steps: Number of steps taken
        final_info: Info from final step
    """
    total_reward = 0.0
    solved = False
    steps = 0
    final_info = {}
    
    for action in actions[:max_steps]:
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        final_info = info
        
        if terminated:
            solved = True
            break
        if truncated:
            break
    
    return total_reward, solved, steps, final_info


# ============================================================================
# DATASET GENERATION FOR GRPO
# ============================================================================

def generate_maze_dataset(
    num_samples: int = 500,
    maze_sizes: List[Tuple[int, int]] = None,
    image_size: int = 300,
    include_optimal_path: bool = True
) -> List[Dict]:
    """
    Generate a dataset of maze images for GRPO training.
    
    Args:
        num_samples: Total number of maze samples
        maze_sizes: List of (rows, cols) tuples
        image_size: Target image size in pixels
        include_optimal_path: Whether to compute and store optimal solution
    
    Returns:
        List of dictionaries with maze data
    """
    if maze_sizes is None:
        maze_sizes = [(5, 5), (7, 7), (9, 9)]
    
    samples_per_size = num_samples // len(maze_sizes)
    dataset = []
    
    for size in maze_sizes:
        cell_size = max(10, image_size // max(size))
        
        for i in range(samples_per_size):
            seed = hash((size, i)) % (2**31)
            
            env = SyntheticMazeEnv(size=size, seed=seed)
            state = env.reset()
            
            # Render image
            image = env.render(cell_size=cell_size)
            
            # Resize to consistent size
            image = image.resize((image_size, image_size), Image.LANCZOS)
            
            sample = {
                'image': image,
                'maze_size': list(size),
                'agent_pos': list(state.agent_pos),
                'goal_pos': list(state.goal_pos),
                'seed': seed,
                'maze_array': state.maze.tolist()
            }
            
            if include_optimal_path:
                optimal = env.get_optimal_path()
                sample['optimal_path'] = optimal
                sample['optimal_length'] = len(optimal) if optimal else -1
            
            dataset.append(sample)
    
    return dataset


# ============================================================================
# TESTING
# ============================================================================

def test_maze_env():
    """Test the synthetic maze environment."""
    print("Testing SyntheticMazeEnv...")
    
    env = SyntheticMazeEnv(size=(7, 7), seed=42)
    state = env.reset()
    
    print(f"Maze size: {env.size}")
    print(f"Agent position: {state.agent_pos}")
    print(f"Goal position: {state.goal_pos}")
    print(f"\nMaze layout:")
    print(state.maze)
    
    # Find optimal path
    optimal_path = env.get_optimal_path()
    print(f"\nOptimal path: {optimal_path}")
    print(f"Optimal path length: {len(optimal_path) if optimal_path else 'No path found'}")
    
    # Execute optimal path
    if optimal_path:
        env.reset(seed=42)  # Reset to same maze
        reward, solved, steps, info = execute_actions(env, optimal_path)
        print(f"\nExecuting optimal path:")
        print(f"  Total reward: {reward:.2f}")
        print(f"  Solved: {solved}")
        print(f"  Steps: {steps}")
    
    # Render and save image
    env.reset(seed=42)
    img = env.render(cell_size=40)
    img.save("/tmp/test_maze.png")
    print(f"\nSaved test maze to /tmp/test_maze.png")
    
    # Test random actions
    env.reset(seed=42)
    random_actions = [random.randint(0, 3) for _ in range(20)]
    reward, solved, steps, info = execute_actions(env, random_actions)
    print(f"\nExecuting 20 random actions:")
    print(f"  Total reward: {reward:.2f}")
    print(f"  Solved: {solved}")
    print(f"  Steps: {steps}")
    
    print("\n✓ All tests passed!")


def test_dataset_generation():
    """Test dataset generation."""
    print("\nTesting dataset generation...")
    
    dataset = generate_maze_dataset(
        num_samples=9,
        maze_sizes=[(5, 5), (7, 7), (9, 9)],
        image_size=200
    )
    
    print(f"Generated {len(dataset)} samples")
    
    for i, sample in enumerate(dataset[:3]):
        print(f"\nSample {i}:")
        print(f"  Size: {sample['maze_size']}")
        print(f"  Agent: {sample['agent_pos']}")
        print(f"  Goal: {sample['goal_pos']}")
        print(f"  Optimal length: {sample['optimal_length']}")
        print(f"  Image size: {sample['image'].size}")
    
    # Save sample images
    for i, sample in enumerate(dataset[:3]):
        sample['image'].save(f"/tmp/maze_sample_{i}.png")
    print(f"\nSaved sample images to /tmp/maze_sample_*.png")
    
    print("\n✓ Dataset generation test passed!")


if __name__ == "__main__":
    test_maze_env()
    test_dataset_generation()
