#!/usr/bin/env python3
"""
Generate random mazes exactly as they appear in PPO training.

This script uses the same environment setup as the training script:
- CustomMazeEnv with the same parameters
- Same wrappers (MazeActionWrapper, DenseRewardWrapper, MazeVisualizationWrapper)
- Same maze generation algorithm
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2c_ppo_acktr.custom_maze_env import CustomMazeEnv
from a2c_ppo_acktr.maze_utils import (
    MazeActionWrapper,
    DenseRewardWrapper,
    MazeVisualizationWrapper
)

def create_training_env(maze_size=5, seed=None):
    """
    Create an environment exactly as used in training.
    This matches the setup in envs.py make_env() function.
    """
    # Create base environment
    env = CustomMazeEnv(width=maze_size, height=maze_size, seed=seed)
    
    # Apply wrappers in the same order as training
    env = MazeActionWrapper(env)
    env = DenseRewardWrapper(env, maze_size=maze_size)
    env = MazeVisualizationWrapper(env, cell_size=None, max_image_size=300)
    
    return env

def generate_maze_image(env, output_path):
    """Generate and save a single maze image."""
    obs, info = env.reset()
    
    # obs is the rendered image from MazeVisualizationWrapper
    if isinstance(obs, np.ndarray):
        # Convert to PIL Image and save
        from PIL import Image
        img = Image.fromarray(obs)
        img.save(output_path)
        print(f"Saved: {output_path}")
    else:
        print(f"Warning: Observation is not a numpy array: {type(obs)}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate random mazes exactly as used in PPO training"
    )
    parser.add_argument(
        "--num-mazes",
        type=int,
        default=20,
        help="Number of mazes to generate (default: 20)"
    )
    parser.add_argument(
        "--maze-size",
        type=int,
        default=5,
        help="Maze size (default: 5 for 5x5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_mazes",
        help="Output directory for maze images (default: training_mazes)"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Starting seed for maze generation (default: 0)"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_mazes} {args.maze_size}x{args.maze_size} mazes...")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Using same environment setup as PPO training\n")
    
    for i in range(args.num_mazes):
        seed = args.start_seed + i
        env = create_training_env(maze_size=args.maze_size, seed=seed)
        
        output_path = os.path.join(args.output_dir, f"maze_{args.maze_size}x{args.maze_size}_seed_{seed:04d}.png")
        generate_maze_image(env, output_path)
        
        env.close()
    
    print(f"\nGenerated {args.num_mazes} mazes in {args.output_dir}/")

if __name__ == "__main__":
    main()

