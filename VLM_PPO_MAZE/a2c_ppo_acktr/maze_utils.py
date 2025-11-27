import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
import io


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
        
        # Track visited cells for visualization
        self.visited_cells = set()
        
        # Update observation space to RGB image
        height = self.maze_size[0] * self.cell_size
        width = self.maze_size[1] * self.cell_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        
    def _get_maze_structure(self):
        # Try to get maze structure from environment
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
        maze_structure = self._get_maze_structure()
        
        # Draw walls if we have maze structure
        if maze_structure is not None:
            for i in range(self.maze_size[0]):
                for j in range(self.maze_size[1]):
                    x1 = j * self.cell_size
                    y1 = i * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    if maze_structure[i][j] == 1: 
                        draw.rectangle([x1, y1, x2, y2], fill=self.wall_color)
        
        for (row, col) in self.visited_cells:
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            draw.rectangle([x1, y1, x2, y2], fill=self.visited_color)
        
        goal_x1 = goal_pos[1] * self.cell_size
        goal_y1 = goal_pos[0] * self.cell_size
        goal_x2 = goal_x1 + self.cell_size
        goal_y2 = goal_y1 + self.cell_size
        draw.ellipse([goal_x1 + 5, goal_y1 + 5, goal_x2 - 5, goal_y2 - 5], 
                    fill=self.goal_color)
        
        # Draw agent
        agent_x1 = agent_pos[1] * self.cell_size
        agent_y1 = agent_pos[0] * self.cell_size
        agent_x2 = agent_x1 + self.cell_size
        agent_y2 = agent_y1 + self.cell_size
        draw.ellipse([agent_x1 + 5, agent_y1 + 5, agent_x2 - 5, agent_y2 - 5], 
                    fill=self.agent_color)
        
        # Draw grid lines
        for i in range(self.maze_size[0] + 1):
            y = i * self.cell_size
            draw.line([(0, y), (width, y)], fill=(128, 128, 128), width=1)
        for j in range(self.maze_size[1] + 1):
            x = j * self.cell_size
            draw.line([(x, 0), (x, height)], fill=(128, 128, 128), width=1)
        
        # Convert to numpy array
        img_array = np.array(img)
        return img_array
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.visited_cells = set()
        
        # Extract agent and goal positions
        agent_pos = self._extract_agent_position(obs, info)
        goal_pos = self._extract_goal_position(obs, info)
        
        # Add starting position to visited
        if agent_pos is not None:
            self.visited_cells.add(tuple(agent_pos))
        
        # Render maze image
        visual_obs = self._render_maze_image(agent_pos, goal_pos)
        
        # Store positions in info for reward computation
        info['agent_pos'] = agent_pos
        info['goal_pos'] = goal_pos
        info['previous_pos'] = agent_pos
        
        return visual_obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract positions
        agent_pos = self._extract_agent_position(obs, info)
        goal_pos = self._extract_goal_position(obs, info)
        previous_pos = info.get('previous_pos', agent_pos)
        
        # Add current position to visited
        if agent_pos is not None:
            self.visited_cells.add(tuple(agent_pos))
        
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
        
        # Try to get from environment
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if hasattr(env, 'goal'):
            goal = env.goal
            if isinstance(goal, (list, tuple, np.ndarray)) and len(goal) >= 2:
                return list(goal[:2])
        elif hasattr(env, 'unwrapped'):
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'goal'):
                goal = unwrapped.goal
                if isinstance(goal, (list, tuple, np.ndarray)) and len(goal) >= 2:
                    return list(goal[:2])
        return None


def compute_dense_reward(previous_pos, current_pos, goal_pos):
    if previous_pos is None or current_pos is None or goal_pos is None:
        return 0.0
    
    # Convert to numpy arrays for easier computation
    prev_pos = np.array(previous_pos)
    curr_pos = np.array(current_pos)
    goal = np.array(goal_pos)
    
    # Compute euclidean distances
    prev_distance = np.linalg.norm(prev_pos - goal)
    curr_distance = np.linalg.norm(curr_pos - goal)
    
    # Reward is the reduction in distance
    reward = prev_distance - curr_distance
    
    # Optional: Add small bonus for reaching goal
    if curr_distance < 0.5:  # Very close to goal
        reward += 1.0
    
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
        obs, info = self.env.reset(**kwargs)
        
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
            # Try to get from environment
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'goal'):
                goal = env.goal
                if isinstance(goal, (list, tuple, np.ndarray)) and len(goal) >= 2:
                    self.goal_pos = list(goal[:2])
        
        # Store in info for visualization wrapper
        if isinstance(info, list):
            for i, inf in enumerate(info):
                if isinstance(inf, dict):
                    inf['previous_pos'] = self.previous_pos
                    inf['goal_pos'] = self.goal_pos
        elif isinstance(info, dict):
            info['previous_pos'] = self.previous_pos
            info['goal_pos'] = self.goal_pos
        
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
            
            # Compute dense reward
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
            
            # Compute dense reward
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

