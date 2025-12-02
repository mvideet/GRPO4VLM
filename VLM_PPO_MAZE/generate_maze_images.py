#!/usr/bin/env python3
"""
Generate random 5x5 mazes and save them as images.

- Uses a standard DFS "perfect maze" generator.
- Draws walls on a 5x5 grid.
- Marks start (top-left) in green and goal (bottom-right) in red.
"""

import os
import random
from PIL import Image, ImageDraw

# Bitmask for walls in each cell
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


def maze_to_image(grid, cell_size=60, wall_thickness=4):
    """
    Convert a maze grid (from generate_maze) into a PIL Image.
    """
    height = len(grid)
    width = len(grid[0])

    img_w = width * cell_size + wall_thickness
    img_h = height * cell_size + wall_thickness
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    # Draw walls for each cell
    for y in range(height):
        for x in range(width):
            cell = grid[y][x]
            x0 = x * cell_size
            y0 = y * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            if cell & N:
                draw.line((x0, y0, x1, y0), fill="black", width=wall_thickness)
            if cell & S:
                draw.line((x0, y1, x1, y1), fill="black", width=wall_thickness)
            if cell & W:
                draw.line((x0, y0, x0, y1), fill="black", width=wall_thickness)
            if cell & E:
                draw.line((x1, y0, x1, y1), fill="black", width=wall_thickness)

    # Draw start (top-left) in green
    start_x = cell_size // 2
    start_y = cell_size // 2
    r = cell_size // 3
    draw.ellipse(
        (start_x - r, start_y - r, start_x + r, start_y + r),
        fill="lime"
    )

    # Draw goal (bottom-right) in red
    goal_x = (width - 1) * cell_size + cell_size // 2
    goal_y = (height - 1) * cell_size + cell_size // 2
    draw.ellipse(
        (goal_x - r, goal_y - r, goal_x + r, goal_y + r),
        fill="red"
    )

    return img


def main(num_mazes=10, out_dir="mazes_5x5"):
    os.makedirs(out_dir, exist_ok=True)

    for i in range(num_mazes):
        # optional: fix or vary the seed if you want reproducibility
        rng = random.Random()  # fresh RNG → different maze each time
        grid = generate_maze(5, 5, rng=rng)
        img = maze_to_image(grid)
        img.save(os.path.join(out_dir, f"maze_{i:03d}.png"))

    print(f"Saved {num_mazes} mazes to {out_dir}/")


if __name__ == "__main__":
    main(num_mazes=20)



