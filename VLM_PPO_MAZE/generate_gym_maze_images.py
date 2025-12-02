#!/usr/bin/env python3
"""
Utility script to generate and save images directly from the gym-maze
environment using the same wrappers as training.

Images are saved into a configurable folder (default: ../gym-maze-imgs).
Each image corresponds to one random 5x5 maze with the agent (green) and
goal (red) rendered using MazeVisualizationWrapper.
"""

import os
import argparse

from generate_maze_images import generate_maze_image


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from the gym-maze environment"
    )
    parser.add_argument(
        "--num-mazes",
        type=int,
        default=20,
        help="Number of mazes/images to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../gym-maze-imgs",
        help="Directory to save images",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Starting seed for maze generation",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(
        f"Generating {args.num_mazes} gym-maze images into {os.path.abspath(args.output_dir)}"
    )
    for i in range(args.num_mazes):
        seed = args.start_seed + i
        generate_maze_image(seed=seed, output_dir=args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()



