# -*- coding: utf-8 -*-
"""
Maze Dataset Generation for GRPO Training
"""

import random
import numpy as np
from PIL import Image
from typing import Dict, Tuple
from datasets import Dataset
from config import MazeGRPOConfig
from maze_env import (
    SyntheticMazeGymAdapter,
    MazeActionWrapper,
    DenseRewardWrapper,
    MazeVisualizationWrapper
)


class MazeDatasetGenerator:
    """Generates maze images and metadata for GRPO training."""
    
    def __init__(self, config: MazeGRPOConfig):
        self.config = config
        self.env_cache = {}
    
    def create_wrapped_env(self, maze_size: Tuple[int, int], seed: int = None):
        """Create a fully wrapped maze environment using synthetic maze."""
        # Create base synthetic environment with gym adapter
        env = SyntheticMazeGymAdapter(maze_size=maze_size, seed=seed, max_image_size=self.config.image_size)
        
        # Apply wrappers in order
        # Note: SyntheticMazeGymAdapter already provides images and dense rewards,
        # but we keep wrappers for compatibility and to ensure consistent interfaces
        env = MazeActionWrapper(env)
        # DenseRewardWrapper: Note that synthetic env already computes dense rewards,
        # but this wrapper adds additional reward shaping if needed
        env = DenseRewardWrapper(env, maze_size=maze_size)
        # Visualization wrapper: synthetic env already renders, but this ensures
        # consistent image format and adds visited cell tracking
        env = MazeVisualizationWrapper(env, max_image_size=self.config.image_size)
        
        return env
    
    def generate_single_maze(self, maze_size: Tuple[int, int], seed: int = None) -> Dict:
        """Generate a single maze instance with image and metadata."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Create environment with seed
        env = self.create_wrapped_env(maze_size, seed=seed)
        result = env.reset(seed=seed)
        
        # Handle both tuple and single return value
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Get positions from info or last_info
        if isinstance(info, dict) and 'agent_pos' in info:
            agent_pos = info['agent_pos']
            goal_pos = info['goal_pos']
        else:
            agent_pos = env._last_info.get('agent_pos', [0, 0])
            goal_pos = env._last_info.get('goal_pos', [maze_size[0]-1, maze_size[1]-1])
        
        # Convert observation to PIL Image
        if isinstance(obs, np.ndarray):
            image = Image.fromarray(obs.astype(np.uint8))
        else:
            image = obs
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return {
            'image': image,
            'maze_size': maze_size,
            'agent_pos': agent_pos,
            'goal_pos': goal_pos,
            'env': env,  # Keep reference for reward computation
            'seed': seed
        }
    
    def generate_dataset(self) -> Dataset:
        """Generate full training dataset."""
        data = []
        
        for difficulty_level, maze_size in enumerate(self.config.maze_sizes):
            for i in range(self.config.num_mazes_per_size):
                seed = hash((maze_size, i)) % (2**32)
                maze_data = self.generate_single_maze(maze_size, seed)
                
                # Store env reference separately (can't serialize to HF dataset)
                # We'll recreate envs during reward computation
                # Ensure all list fields are consistently lists
                agent_pos = maze_data['agent_pos']
                goal_pos = maze_data['goal_pos']
                data.append({
                    'image': maze_data['image'],
                    'maze_size': list(maze_data['maze_size']),
                    'agent_pos': list(agent_pos) if not isinstance(agent_pos, list) else agent_pos,
                    'goal_pos': list(goal_pos) if not isinstance(goal_pos, list) else goal_pos,
                    'seed': int(seed),
                    'difficulty_level': difficulty_level  # Add difficulty level for curriculum learning
                })
        
        return Dataset.from_list(data)
    
    def save_maze_images(self, dataset: Dataset, output_dir: str = "maze_images"):
        """Save all maze images from the dataset to disk."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n  Saving maze images to {output_dir}/...")
        
        for idx, example in enumerate(dataset):
            maze_size = example['maze_size']
            seed = example['seed']
            agent_pos = example['agent_pos']
            goal_pos = example['goal_pos']
            
            # Create subdirectory for each maze size
            size_dir = os.path.join(output_dir, f"{maze_size[0]}x{maze_size[1]}")
            os.makedirs(size_dir, exist_ok=True)
            
            # Create filename with seed and positions
            filename = f"maze_{maze_size[0]}x{maze_size[1]}_seed{seed}_start{agent_pos[0]},{agent_pos[1]}_goal{goal_pos[0]},{goal_pos[1]}.png"
            filepath = os.path.join(size_dir, filename)
            
            # Save image
            image = example['image']
            if isinstance(image, Image.Image):
                image.save(filepath)
            elif isinstance(image, np.ndarray):
                Image.fromarray(image.astype(np.uint8)).save(filepath)
            else:
                # Try to convert if it's some other format
                Image.fromarray(np.array(image)).save(filepath)
        
        print(f"  Saved {len(dataset)} maze images to {output_dir}/")

