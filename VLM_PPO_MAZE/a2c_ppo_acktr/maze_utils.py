import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
import io

def _unwrap_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


class MazeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # We want 4 discrete actions for the agent/policy:
        # 0 -> 'N', 1 -> 'S', 2 -> 'E', 3 -> 'W'
        self.action_space = gym.spaces.Discrete(4)
        # Check if this is a custom maze env (uses integer actions directly)
        self._is_custom_maze = self._check_custom_maze()

    def _check_custom_maze(self):
        """Check if wrapped env is CustomMazeEnv."""
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, '__class__') and 'CustomMazeEnv' in str(env.__class__):
                return True
            env = env.env
        if hasattr(env, '__class__') and 'CustomMazeEnv' in str(env.__class__):
            return True
        return False

    def _base_env(self):
        base = self.env
        while hasattr(base, "env"):
            base = base.env
        return base

    def reset(self, **kwargs):
        base = self._base_env()
        result = base.reset(**kwargs)
        
        # CustomMazeEnv already returns (obs, info) tuple
        if isinstance(result, tuple) and len(result) == 2:
            return result
        # Old gym-maze format
        return result, {}

    def step(self, action):
        """
        Handle actions for both CustomMazeEnv (integer) and gym-maze (string).
        """
        # Convert to a plain Python int
        if isinstance(action, np.ndarray):
            action_idx = int(action.item())
        else:
            action_idx = int(action)

        base = self._base_env()
        
        # CustomMazeEnv uses integer actions directly
        if self._is_custom_maze:
            result = base.step(action_idx)
            # CustomMazeEnv returns 5-tuple: (obs, reward, terminated, truncated, info)
            if isinstance(result, tuple) and len(result) == 5:
                return result
            elif isinstance(result, tuple) and len(result) == 4:
                obs, reward, done, info = result
                return obs, reward, bool(done), False, info
        
        # Old gym-maze format: convert to string direction
        direction_map = {0: "N", 1: "S", 2: "E", 3: "W"}
        direction = direction_map.get(action_idx, "N")
        result = base.step(direction)
        
        # Handle old 4-tuple format
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        elif isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            raise RuntimeError(
                f"Unexpected step output: {type(result)} with len="
                f"{len(result) if isinstance(result, tuple) else 'N/A'}"
            )

        return obs, reward, terminated, truncated, info



class MazeVisualizationWrapper(gym.Wrapper):
    
    def __init__(self, env, cell_size=None, wall_color=(0, 0, 0), 
                 path_color=(255, 255, 255), agent_color=(0, 255, 0),
                 goal_color=(255, 0, 0), visited_color=(200, 200, 200), max_image_size=300):
        super().__init__(env)
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
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        
    def _get_maze_structure(self):
        if hasattr(self.env, 'maze_structure'):
            return self.env.maze_structure
        elif hasattr(self.env.unwrapped, 'maze_structure'):
            return self.env.unwrapped.maze_structure
        else:
            return None
    
    def _render_maze_image(self, agent_pos, goal_pos):
        height = self.maze_size[0] * self.cell_size
        width = self.maze_size[1] * self.cell_size
        
        img = Image.new('RGB', (width, height), color=self.path_color)
        draw = ImageDraw.Draw(img)
        
        # Try to get maze grid (bitmask format) from custom maze env
        maze_grid = self._get_maze_grid()
        
        if maze_grid is not None:
            # Render walls from bitmask format (like generate_maze_images.py)
            # Bitmask constants
            N, S, E, W = 1, 2, 4, 8
            wall_thickness = 2  # Thin walls like in the example
            
            for y in range(self.maze_size[0]):
                for x in range(self.maze_size[1]):
                    cell = maze_grid[y][x]
                    x0 = x * self.cell_size
                    y0 = y * self.cell_size
                    x1 = x0 + self.cell_size
                    y1 = y0 + self.cell_size
                    
                    # Draw walls based on bitmask
                    if cell & N:
                        draw.line((x0, y0, x1, y0), fill=self.wall_color, width=wall_thickness)
                    if cell & S:
                        draw.line((x0, y1, x1, y1), fill=self.wall_color, width=wall_thickness)
                    if cell & W:
                        draw.line((x0, y0, x0, y1), fill=self.wall_color, width=wall_thickness)
                    if cell & E:
                        draw.line((x1, y0, x1, y1), fill=self.wall_color, width=wall_thickness)
        else:
            # Fallback: try old maze_structure format (for compatibility)
            maze_structure = self._get_maze_structure()
            if maze_structure is not None:
                try:
                    maze_arr = np.array(maze_structure)
                    for i in range(self.maze_size[0]):
                        for j in range(self.maze_size[1]):
                            x1 = j * self.cell_size
                            y1 = i * self.cell_size
                            x2 = x1 + self.cell_size
                            y2 = y1 + self.cell_size
                            if maze_arr[i][j] == 1: 
                                draw.rectangle([x1, y1, x2, y2], fill=self.wall_color)
                except Exception as e:
                    pass
        
        # Draw visited cells
        for (row, col) in self.visited_cells:
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            draw.rectangle([x1, y1, x2, y2], fill=self.visited_color)
        
        # Draw grid lines (light gray, thin)
        for i in range(self.maze_size[0] + 1):
            y = i * self.cell_size
            draw.line([(0, y), (width, y)], fill=(128, 128, 128), width=1)
        for j in range(self.maze_size[1] + 1):
            x = j * self.cell_size
            draw.line([(x, 0), (x, height)], fill=(128, 128, 128), width=1)
        
        # Draw goal (red circle) at bottom-right
        goal_x1 = goal_pos[1] * self.cell_size + 5
        goal_y1 = goal_pos[0] * self.cell_size + 5
        goal_x2 = goal_x1 + self.cell_size - 10
        goal_y2 = goal_y1 + self.cell_size - 10
        draw.ellipse([goal_x1, goal_y1, goal_x2, goal_y2], fill=self.goal_color)
        
        # Draw agent (green circle) at current position
        agent_x1 = agent_pos[1] * self.cell_size + 5
        agent_y1 = agent_pos[0] * self.cell_size + 5
        agent_x2 = agent_x1 + self.cell_size - 10
        agent_y2 = agent_y1 + self.cell_size - 10
        draw.ellipse([agent_x1, agent_y1, agent_x2, agent_y2], fill=self.agent_color)
        
        # Convert to numpy array
        img_array = np.array(img)
        return img_array
    
    def _get_maze_grid(self):
        """Get maze grid in bitmask format from custom maze env."""
        # Check cached grid from reset
        if hasattr(self, '_cached_maze_grid'):
            return self._cached_maze_grid
        
        # Try to get from wrapped environment
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, 'get_maze_grid'):
                grid = env.get_maze_grid()
                if grid is not None:
                    return grid
            env = env.env
        
        # Try direct access
        if hasattr(env, 'get_maze_grid'):
            grid = env.get_maze_grid()
            if grid is not None:
                return grid
        if hasattr(env, 'maze_grid'):
            return env.maze_grid
        
        return None
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        self.visited_cells = set()
        agent_pos = self._extract_agent_position(obs, info)
        goal_pos = self._extract_goal_position(obs, info)
        if agent_pos is not None:
            self.visited_cells.add(tuple(agent_pos))
        
        # Store maze grid reference if available
        if isinstance(info, dict) and 'maze_grid' in info:
            self._cached_maze_grid = info['maze_grid']
        
        # Also try to get from environment directly
        if not hasattr(self, '_cached_maze_grid'):
            maze_grid = self._get_maze_grid()
            if maze_grid is not None:
                self._cached_maze_grid = maze_grid

        # Render maze image
        visual_obs = self._render_maze_image(agent_pos, goal_pos)
        if isinstance(info, dict):
            info_out = dict(info)
        else:
            info_out = {}

        info_out['agent_pos'] = agent_pos
        info_out['goal_pos'] = goal_pos
        info_out['previous_pos'] = agent_pos

        return visual_obs, info_out
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract positions
        agent_pos = self._extract_agent_position(obs, info)
        goal_pos = self._extract_goal_position(obs, info)
        previous_pos = info.get('previous_pos', agent_pos)
        
        # Add current position to visited
        if agent_pos is not None:
            self.visited_cells.add(tuple(agent_pos))
        
        # Update cached maze grid if available in info
        if isinstance(info, dict) and 'maze_grid' in info:
            self._cached_maze_grid = info['maze_grid']
        
        # Render maze image
        visual_obs = self._render_maze_image(agent_pos, goal_pos)
        
        # Store positions in info
        info['agent_pos'] = agent_pos
        info['goal_pos'] = goal_pos
        info['previous_pos'] = agent_pos
        
        return visual_obs, reward, terminated, truncated, info
    
    def _extract_agent_position(self, obs, info):
        """Extract agent position from observation or info."""
        # Try multiple ways to get position
        if isinstance(info, dict):
            if 'agent_pos' in info:
                return info['agent_pos']
            elif 'position' in info:
                return info['position']
        elif isinstance(info, list) and len(info) > 0:
            # Vectorized environment returns list of infos
            info_dict = info[0] if isinstance(info[0], dict) else {}
            if 'agent_pos' in info_dict:
                return info_dict['agent_pos']
            elif 'position' in info_dict:
                return info_dict['position']
        
        # Try to get from observation (gym-maze typically returns [row, col])
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1 and obs.size >= 2:
                return obs[:2].tolist()
            elif obs.ndim == 3:
                # This is an image observation, try to get from environment
                pass
        
        # Fallback: try to get from environment
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if hasattr(env, 'state'):
            state = env.state
            if isinstance(state, (list, tuple, np.ndarray)) and len(state) >= 2:
                return list(state[:2])
        elif hasattr(env, 'observation_space'):
            # Try to get from unwrapped environment
            if hasattr(env, 'unwrapped'):
                unwrapped = env.unwrapped
                if hasattr(unwrapped, 'state'):
                    state = unwrapped.state
                    if isinstance(state, (list, tuple, np.ndarray)) and len(state) >= 2:
                        return list(state[:2])
        return None
    
    def _extract_goal_position(self, obs, info):
        """Extract goal position from observation or info."""
        if isinstance(info, dict):
            if 'goal_pos' in info:
                return info['goal_pos']
            elif 'goal' in info:
                return info['goal']
        elif isinstance(info, list) and len(info) > 0:
            # Vectorized environment returns list of infos
            info_dict = info[0] if isinstance(info[0], dict) else {}
            if 'goal_pos' in info_dict:
                return info_dict['goal_pos']
            elif 'goal' in info_dict:
                return info_dict['goal']
        
        base = _unwrap_env(self.env)

        # In gym-maze, MazeEnv has a .maze_view attribute which is a MazeView2D
        mv = getattr(base, "maze_view", None)
        if mv is not None and hasattr(mv, "goal"):
            goal = mv.goal  # e.g. numpy array [x, y]
            if isinstance(goal, np.ndarray):
                goal = goal.tolist()
            # Ensure it's a simple [row, col] / [x, y]
            if isinstance(goal, (list, tuple)) and len(goal) >= 2:
                return list(goal[:2])

        # 3) As a *very* last resort, infer from maze size (still derived from env)
        # (this is logically bottom-right, but derived from env.maze_size, not a magic constant)
        if mv is not None and hasattr(mv, "maze"):
            maze = mv.maze  # Maze object
            if hasattr(maze, "maze_size"):
                size = maze.maze_size
                if isinstance(size, (list, tuple)) and len(size) >= 2:
                    # bottom right cell
                    return [size[0] - 1, size[1] - 1]

        return None


def compute_dense_reward(previous_pos, current_pos, goal_pos):
    if current_pos is None or goal_pos is None:
        return 0.0

    curr_pos = np.array(current_pos)
    goal = np.array(goal_pos)
    curr_distance = np.linalg.norm(curr_pos - goal)
    
    # Check if reached goal: agent is in same cell as goal (discrete maze)
    # Convert to integer cell coordinates for robust comparison
    agent_cell = (int(round(curr_pos[0])), int(round(curr_pos[1])))
    goal_cell = (int(round(goal[0])), int(round(goal[1])))
    reached_goal = (agent_cell == goal_cell)
    
    # Compute reward: delta distance (positive if moving closer)
    if previous_pos is not None:
        prev_pos = np.array(previous_pos)
        prev_distance = np.linalg.norm(prev_pos - goal)
        reward = prev_distance - curr_distance  # Positive if moving closer
    else:
        # First step: no previous position, use negative distance as baseline
        reward = -curr_distance
    
    # Add goal bonus if reached goal
    if reached_goal:
        goal_bonus = 5
        reward += goal_bonus
    
    return float(reward)


class DenseRewardWrapper(gym.Wrapper):
    def __init__(self, env, maze_size=None):
        super().__init__(env)
        self.previous_pos = None
        self.goal_pos = None
        # Store maze size for visualization wrapper
        if maze_size is not None:
            self.maze_size = maze_size if isinstance(maze_size, (tuple, list)) else (maze_size, maze_size)
        else:
            # Try to infer from environment
            unwrapped = env
            while hasattr(unwrapped, 'env'):
                unwrapped = unwrapped.env
            if hasattr(unwrapped, 'maze_size'):
                size = unwrapped.maze_size
                self.maze_size = size if isinstance(size, (tuple, list)) else (size, size)
            else:
                self.maze_size = (5, 5)  # default
    
    def reset(self, **kwargs):
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        
        # Handle vectorized environments (info is a list)
        if isinstance(info, list) and len(info) > 0:
            info_dict = info[0] if isinstance(info[0], dict) else {}
        else:
            info_dict = info if isinstance(info, dict) else {}
        
        # Extract positions
        if 'agent_pos' in info_dict:
            self.previous_pos = info_dict['agent_pos']
        elif 'position' in info_dict:
            self.previous_pos = info_dict['position']
        else:
            # Try to extract from observation
            if isinstance(obs, np.ndarray):
                if obs.ndim == 1 and obs.size >= 2:
                    self.previous_pos = obs[:2].tolist()
                elif obs.ndim == 3:
                    # Image observation, try to get from environment
                    env = self.env
                    while hasattr(env, 'env'):
                        env = env.env
                    if hasattr(env, 'state'):
                        state = env.state
                        if isinstance(state, (list, tuple, np.ndarray)) and len(state) >= 2:
                            self.previous_pos = list(state[:2])
        
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

            if self.goal_pos is None and mv is not None and hasattr(mv, "maze"):
                maze = mv.maze
                if hasattr(maze, "maze_size"):
                    size = maze.maze_size
                    if isinstance(size, (list, tuple)) and len(size) >= 2:
                        self.goal_pos = [size[0] - 1, size[1] - 1]
        
        # Store in info for visualization wrapper
        if isinstance(info, list):
            for i, inf in enumerate(info):
                if isinstance(inf, dict):
                    inf['previous_pos'] = self.previous_pos
                    inf['goal_pos'] = self.goal_pos
        elif isinstance(info, dict):
            info['previous_pos'] = self.previous_pos
            info['goal_pos'] = self.goal_pos
        else:
            info = {'previous_pos': self.previous_pos, 'goal_pos': self.goal_pos}
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Handle vectorized environments
        if isinstance(info, list):
            # For vectorized envs, process first environment
            info_dict = info[0] if isinstance(info[0], dict) else {}
            current_pos = None
            
            if 'agent_pos' in info_dict:
                current_pos = info_dict['agent_pos']
            elif 'position' in info_dict:
                current_pos = info_dict['position']
            else:
                # Try to extract from observation
                if isinstance(obs, np.ndarray):
                    if obs.ndim == 1 and obs.size >= 2:
                        current_pos = obs[:2].tolist()
                    elif obs.ndim == 2 and obs.shape[0] >= 1:
                        # Batch of observations
                        if obs.shape[1] >= 2:
                            current_pos = obs[0, :2].tolist()
            
            if current_pos is None:
                current_pos = self.previous_pos
            
            # Compute dense reward (delta distance: prev_dist - curr_dist)
            dense_reward = compute_dense_reward(
                self.previous_pos, current_pos, self.goal_pos
            )
            
            # Update previous position
            self.previous_pos = current_pos
            
            # Store in info
            for i, inf in enumerate(info):
                if isinstance(inf, dict):
                    inf['agent_pos'] = current_pos
                    inf['goal_pos'] = self.goal_pos
                    inf['previous_pos'] = self.previous_pos
            
            # Convert reward to array if needed
            if isinstance(reward, (int, float)):
                reward = np.array([reward])
            elif isinstance(reward, np.ndarray) and reward.ndim == 0:
                reward = np.array([reward])
            
            return obs, dense_reward, terminated, truncated, info
        else:
            # Single environment
            info_dict = info if isinstance(info, dict) else {}
            current_pos = None
            
            if 'agent_pos' in info_dict:
                current_pos = info_dict['agent_pos']
            elif 'position' in info_dict:
                current_pos = info_dict['position']
            else:
                # Try to extract from observation
                if isinstance(obs, np.ndarray) and obs.size >= 2:
                    if obs.ndim == 1:
                        current_pos = obs[:2].tolist()
            
            if current_pos is None:
                current_pos = self.previous_pos
            
            # Compute dense reward (delta distance: prev_dist - curr_dist)
            dense_reward = compute_dense_reward(
                self.previous_pos, current_pos, self.goal_pos
            )
            
            # Update previous position
            self.previous_pos = current_pos
            
            # Store in info
            if isinstance(info, dict):
                info['agent_pos'] = current_pos
                info['goal_pos'] = self.goal_pos
                info['previous_pos'] = self.previous_pos
            
            return obs, dense_reward, terminated, truncated, info

