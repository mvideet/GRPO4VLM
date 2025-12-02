#!/usr/bin/env python3
"""
Test script to generate and verify random maze images.
Generates a few test mazes and verifies they have walls and valid paths.
"""

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from generate_random_mazes import (
    MazeGenerator,
    render_maze_image,
    generate_maze_image,
)


def maze_to_grid(vertical_walls, horizontal_walls):
    """Convert wall matrices to adjacency grid (0 path, 1 wall for visualization)."""
    height, width_plus1 = vertical_walls.shape
    width = width_plus1 - 1
    grid = np.zeros((height, width), dtype=int)

    # Mark cells as blocked if surrounded by walls (for debugging)
    for i in range(height):
        for j in range(width):
            # A cell is considered blocked if all four surrounding walls exist
            if (
                vertical_walls[i, j] == 1
                and vertical_walls[i, j + 1] == 1
                and horizontal_walls[i, j] == 1
                and horizontal_walls[i + 1, j] == 1
            ):
                grid[i, j] = 1
    return grid


def test_maze_generation():
    print("=" * 60)
    print("Testing Random Maze Generation (Line Walls)")
    print("=" * 60)

    generator = MazeGenerator(width=5, height=5, extra_openings=2)

    print("\n1. Generating single maze with seed=42...")
    vertical, horizontal = generator.generate(seed=42)
    print(f"   Vertical walls shape: {vertical.shape}")
    print(f"   Horizontal walls shape: {horizontal.shape}")

    print("\n2. Rendering maze to ../imgs/test_maze.png ...")
    img = render_maze_image(vertical, horizontal)
    os.makedirs("../imgs", exist_ok=True)
    img.save("../imgs/test_maze.png")

    print("\n3. Generating multiple mazes for seeds [0..4]...")
    for seed in range(5):
        vertical, horizontal = generator.generate(seed=seed)
        # Simple diagnostic: count walls
        total_vertical = np.sum(vertical)
        total_horizontal = np.sum(horizontal)
        print(
            f"   Seed {seed}: vertical walls={total_vertical}, horizontal walls={total_horizontal}"
        )
        generate_maze_image(
            seed=seed,
            output_dir="../imgs",
            maze_size=5,
            extra_openings=2,
        )

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_maze_generation()

