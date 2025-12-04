# -*- coding: utf-8 -*-
"""
Maze Environment Wrappers and Adapters
"""

import numpy as np
import gym
from typing import List, Dict, Any, Tuple
from synthetic_maze import SyntheticMazeEnv


def _unwrap_env(env):
    """Get the base unwrapped environment."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


class SyntheticMazeGymAdapter(gym.Env):
    """
    Adapter to make SyntheticMazeEnv compatible with gym/gymnasium interface.
    Wraps SyntheticMazeEnv to provide gym-like API.
    """
    
    def __init__(self, maze_size: Tuple[int, int], seed: int = None, max_image_size: int = 300):
        super().__init__()
        self.maze_size = maze_size if isinstance(maze_size, (tuple, list)) else (maze_size, maze_size)
        self.synthetic_env = SyntheticMazeEnv(size=self.maze_size, seed=seed)
        self.max_image_size = max_image_size
        max_dim = max(self.maze_size)
        self.cell_size = max(10, int(max_image_size / max_dim))
        height = self.maze_size[0] * self.cell_size
        width = self.maze_size[1] * self.cell_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(4)
        
        self._last_info = {}
    
    def reset(self, seed: int = None, **kwargs):
        """Reset environment and return (observation, info)."""
        state = self.synthetic_env.reset(seed=seed)
        
        # Render image
        image = self.synthetic_env.render(cell_size=self.cell_size)
        
        # Convert to numpy array
        obs = np.array(image)
        
        # Create info dict
        info = {
            'agent_pos': list(state.agent_pos),
            'goal_pos': list(state.goal_pos),
            'maze_structure': state.maze,
            'previous_pos': list(state.agent_pos)
        }
        
        self._last_info = info
        
        return obs, info
    
    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        state, reward, terminated, truncated, info = self.synthetic_env.step(action)
        
        # Render image
        image = self.synthetic_env.render(cell_size=self.cell_size)
        obs = np.array(image)
        
        # Update info with positions
        info['agent_pos'] = list(state.agent_pos)
        info['goal_pos'] = list(state.goal_pos)
        info['maze_structure'] = state.maze
        info['previous_pos'] = info.get('prev_pos', list(state.agent_pos))
        
        self._last_info = info
        
        return obs, reward, terminated, truncated, info
    
    @property
    def maze_structure(self):
        """Get current maze structure."""
        if self.synthetic_env.state is not None:
            return self.synthetic_env.state.maze
        return None


class MazeActionWrapper(gym.ActionWrapper):
    """Converts discrete actions (0-3) to direction strings (N/S/E/W)."""
    
    def __init__(self, env):
        super().__init__(env)
        self._action_map = {0: "N", 1: "S", 2: "E", 3: "W"}
        self.action_space = gym.spaces.Discrete(4)

    def _base_env(self):
        base = self.env
        while hasattr(base, "env"):
            base = base.env
        return base

    def reset(self, **kwargs):
        base = self._base_env()
        result = base.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        self._last_info = info
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action_idx = int(action.item())
        else:
            action_idx = int(action)
        
        # Pass integer action directly (synthetic env expects integers, not strings)
        # The action_map is kept for compatibility but not used
        base = self._base_env()
        result = base.step(action_idx)
        
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        elif isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            raise RuntimeError(f"Unexpected step output: {result}")
        
        return obs, reward, terminated, truncated, info


class MazeVisualizationWrapper(gym.Wrapper):
    """Renders maze state as RGB image observation."""
    
    def __init__(self, env, cell_size=None, wall_color=(0, 0, 0),
                 path_color=(255, 255, 255), agent_color=(0, 255, 0),
                 goal_color=(255, 0, 0), visited_color=(200, 200, 200),
                 max_image_size=300):
        super().__init__(env)
        
        # Get maze size
        size_attr = getattr(env, 'maze_size', None)
        if size_attr is None and hasattr(env, 'unwrapped'):
            size_attr = getattr(env.unwrapped, 'maze_size', None)
        if size_attr is None:
            size_attr = (5, 5)
        if not isinstance(size_attr, (tuple, list)):
            size_attr = (size_attr, size_attr)
        self.maze_size = tuple(size_attr)
        
        if cell_size is None:
            max_dim = max(self.maze_size)
            cell_size = max(2, int(max_image_size / max_dim))
        
        self.cell_size = cell_size
        self.max_image_size = max_image_size
        self.wall_color = wall_color
        self.path_color = path_color
        self.agent_color = agent_color
        self.goal_color = goal_color
        self.visited_color = visited_color
        self.visited_cells = set()
        
        height = self.maze_size[0] * self.cell_size
        width = self.maze_size[1] * self.cell_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

    def _get_maze_structure(self):
        # Try to get maze structure from various sources
        if hasattr(self.env, 'maze_structure'):
            return self.env.maze_structure
        
        # Check unwrapped env
        unwrapped = _unwrap_env(self.env)
        if hasattr(unwrapped, 'maze_structure'):
            return unwrapped.maze_structure
        
        # Check if it's a synthetic env adapter
        if hasattr(unwrapped, 'synthetic_env') and hasattr(unwrapped.synthetic_env, 'state'):
            if unwrapped.synthetic_env.state is not None:
                return unwrapped.synthetic_env.state.maze
        
        # Check for maze_structure in info
        if hasattr(self.env, '_last_info') and 'maze_structure' in self.env._last_info:
            return self.env._last_info['maze_structure']
        
        return None

    def _render_maze_image(self, agent_pos, goal_pos):
        from PIL import Image, ImageDraw
        
        height = self.maze_size[0] * self.cell_size
        width = self.maze_size[1] * self.cell_size
        
        img = Image.new('RGB', (width, height), color=self.path_color)
        draw = ImageDraw.Draw(img)
        
        # Draw walls
        maze_structure = self._get_maze_structure()
        if maze_structure is not None:
            for i in range(self.maze_size[0]):
                for j in range(self.maze_size[1]):
                    x1 = j * self.cell_size
                    y1 = i * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    if maze_structure[i][j] == 1:
                        draw.rectangle([x1, y1, x2, y2], fill=self.wall_color)
        
        # Draw visited cells
        for (row, col) in self.visited_cells:
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            draw.rectangle([x1, y1, x2, y2], fill=self.visited_color)
        
        # Draw goal
        if goal_pos is not None:
            goal_x1 = goal_pos[1] * self.cell_size
            goal_y1 = goal_pos[0] * self.cell_size
            goal_x2 = goal_x1 + self.cell_size
            goal_y2 = goal_y1 + self.cell_size
            padding = max(2, self.cell_size // 6)
            draw.ellipse([goal_x1 + padding, goal_y1 + padding,
                         goal_x2 - padding, goal_y2 - padding],
                        fill=self.goal_color)
        
        # Draw agent
        if agent_pos is not None:
            agent_x1 = agent_pos[1] * self.cell_size
            agent_y1 = agent_pos[0] * self.cell_size
            agent_x2 = agent_x1 + self.cell_size
            agent_y2 = agent_y1 + self.cell_size
            padding = max(2, self.cell_size // 6)
            draw.ellipse([agent_x1 + padding, agent_y1 + padding,
                         agent_x2 - padding, agent_y2 - padding],
                        fill=self.agent_color)
        
        # Draw grid lines
        for i in range(self.maze_size[0] + 1):
            y = i * self.cell_size
            draw.line([(0, y), (width, y)], fill=(128, 128, 128), width=1)
        for j in range(self.maze_size[1] + 1):
            x = j * self.cell_size
            draw.line([(x, 0), (x, height)], fill=(128, 128, 128), width=1)
        
        return np.array(img)

    def _extract_agent_position(self, obs, info):
        if isinstance(info, dict):
            if 'agent_pos' in info:
                pos = info['agent_pos']
                return list(pos) if isinstance(pos, (tuple, list)) else pos
            elif 'position' in info:
                pos = info['position']
                return list(pos) if isinstance(pos, (tuple, list)) else pos
        
        if isinstance(obs, np.ndarray) and obs.ndim == 1 and obs.size >= 2:
            return obs[:2].tolist()
        
        env = _unwrap_env(self.env)
        # Check synthetic env state
        if hasattr(env, 'synthetic_env') and hasattr(env.synthetic_env, 'state'):
            if env.synthetic_env.state is not None:
                return list(env.synthetic_env.state.agent_pos)
        elif hasattr(env, 'state'):
            state = env.state
            if hasattr(state, 'agent_pos'):
                return list(state.agent_pos)
            elif isinstance(state, (list, tuple, np.ndarray)) and len(state) >= 2:
                return list(state[:2])
        return None

    def _extract_goal_position(self, obs, info):
        if isinstance(info, dict):
            if 'goal_pos' in info:
                pos = info['goal_pos']
                return list(pos) if isinstance(pos, (tuple, list)) else pos
            elif 'goal' in info:
                pos = info['goal']
                return list(pos) if isinstance(pos, (tuple, list)) else pos
        
        base = _unwrap_env(self.env)
        # Check synthetic env state
        if hasattr(base, 'synthetic_env') and hasattr(base.synthetic_env, 'state'):
            if base.synthetic_env.state is not None:
                return list(base.synthetic_env.state.goal_pos)
        
        # Legacy gym_maze support
        mv = getattr(base, "maze_view", None)
        if mv is not None and hasattr(mv, "goal"):
            goal = mv.goal
            if isinstance(goal, np.ndarray):
                goal = goal.tolist()
            if isinstance(goal, (list, tuple)) and len(goal) >= 2:
                return list(goal[:2])
        
        # Default: bottom-right corner
        return [self.maze_size[0] - 1, self.maze_size[1] - 1]

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        
        self.visited_cells = set()
        agent_pos = self._extract_agent_position(obs, info)
        goal_pos = self._extract_goal_position(obs, info)
        
        if agent_pos is not None:
            self.visited_cells.add(tuple(agent_pos))
        
        visual_obs = self._render_maze_image(agent_pos, goal_pos)
        
        self._last_info = {
            'agent_pos': agent_pos,
            'goal_pos': goal_pos,
            'previous_pos': agent_pos
        }
        
        return visual_obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        agent_pos = self._extract_agent_position(obs, info)
        goal_pos = self._extract_goal_position(obs, info)
        
        if agent_pos is not None:
            self.visited_cells.add(tuple(agent_pos))
        
        visual_obs = self._render_maze_image(agent_pos, goal_pos)
        
        info['agent_pos'] = agent_pos
        info['goal_pos'] = goal_pos
        info['previous_pos'] = agent_pos
        
        return visual_obs, reward, terminated, truncated, info


def compute_dense_reward(previous_pos, current_pos, goal_pos):
    """Compute reward based on distance improvement toward goal."""
    if previous_pos is None or current_pos is None or goal_pos is None:
        return 0.0
    
    prev_pos = np.array(previous_pos)
    curr_pos = np.array(current_pos)
    goal = np.array(goal_pos)
    
    prev_distance = np.linalg.norm(prev_pos - goal)
    curr_distance = np.linalg.norm(curr_pos - goal)
    
    reward = prev_distance - curr_distance
    
    if curr_distance < 0.5:
        reward += 1.0
    
    return float(reward)


class DenseRewardWrapper(gym.Wrapper):
    """Adds dense reward based on distance to goal."""
    
    def __init__(self, env, maze_size=None):
        super().__init__(env)
        self.previous_pos = None
        self.goal_pos = None
        
        if maze_size is not None:
            self.maze_size = maze_size if isinstance(maze_size, (tuple, list)) else (maze_size, maze_size)
        else:
            unwrapped = _unwrap_env(env)
            if hasattr(unwrapped, 'maze_size'):
                size = unwrapped.maze_size
                self.maze_size = size if isinstance(size, (tuple, list)) else (size, size)
            else:
                self.maze_size = (5, 5)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        
        info_dict = info if isinstance(info, dict) else {}
        
        # Extract agent position
        if 'agent_pos' in info_dict:
            self.previous_pos = info_dict['agent_pos']
        elif 'position' in info_dict:
            self.previous_pos = info_dict['position']
        elif isinstance(obs, np.ndarray) and obs.ndim == 1 and obs.size >= 2:
            self.previous_pos = obs[:2].tolist()
        else:
            env = _unwrap_env(self.env)
            if hasattr(env, 'state'):
                state = env.state
                if isinstance(state, (list, tuple, np.ndarray)) and len(state) >= 2:
                    self.previous_pos = list(state[:2])
        
        # Extract goal position
        if 'goal_pos' in info_dict:
            self.goal_pos = info_dict['goal_pos']
        elif 'goal' in info_dict:
            self.goal_pos = info_dict['goal']
        else:
            base = _unwrap_env(self.env)
            mv = getattr(base, "maze_view", None)
            if mv is not None and hasattr(mv, "goal"):
                goal = mv.goal
                if isinstance(goal, np.ndarray):
                    goal = goal.tolist()
                if isinstance(goal, (list, tuple)) and len(goal) >= 2:
                    self.goal_pos = list(goal[:2])
            
            if self.goal_pos is None:
                self.goal_pos = [self.maze_size[0] - 1, self.maze_size[1] - 1]
        
        self._last_info = {
            'previous_pos': self.previous_pos,
            'goal_pos': self.goal_pos
        }
        
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        info_dict = info if isinstance(info, dict) else {}
        current_pos = None
        
        if 'agent_pos' in info_dict:
            current_pos = info_dict['agent_pos']
        elif 'position' in info_dict:
            current_pos = info_dict['position']
        elif isinstance(obs, np.ndarray) and obs.ndim == 1 and obs.size >= 2:
            current_pos = obs[:2].tolist()
        
        if current_pos is None:
            current_pos = self.previous_pos
        
        dense_reward = compute_dense_reward(
            self.previous_pos, current_pos, self.goal_pos
        )
        
        self.previous_pos = current_pos
        
        if isinstance(info, dict):
            info['agent_pos'] = current_pos
            info['goal_pos'] = self.goal_pos
            info['previous_pos'] = self.previous_pos
        
        return obs, dense_reward, terminated, truncated, info


def grpo_act(env, action_sequence: List[int], max_steps: int = 100) -> Tuple[Any, float, bool, bool, Dict[str, Any], int]:
    """
    Execute a sequence of actions in the environment.
    
    Args:
        env: Gymnasium environment
        action_sequence: List of action indices (0=N, 1=S, 2=E, 3=W)
        max_steps: Maximum steps to execute
    
    Returns:
        Tuple of (final_obs, total_reward, terminated, truncated, final_info, step_count)
    """
    total_reward = 0.0
    terminated = False
    truncated = False
    final_info: Dict[str, Any] = {}
    step_count = 0
    obs = None
    
    for action_idx in action_sequence[:max_steps]:
        if terminated or truncated:
            break
        
        result = env.step(action_idx)
        if len(result) == 4:
            # Legacy gym API (deprecated)
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        elif len(result) == 5:
            # Modern Gymnasium API
            obs, reward, terminated, truncated, info = result
        else:
            raise RuntimeError(f"Unexpected step() return length: {len(result)}, expected 4 or 5")
        
        total_reward += reward
        final_info = info if isinstance(info, dict) else {}
        step_count += 1
    
    return obs, total_reward, terminated, truncated, final_info, step_count

