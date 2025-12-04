# -*- coding: utf-8 -*-
"""
Configuration for Maze GRPO Training
"""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class MazeGRPOConfig:
    """Configuration for maze GRPO training."""
    # Model settings
    model_name: str = "unsloth/gemma-3-4b-it"
    load_in_4bit: bool = True
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    
    # Maze settings
    maze_sizes: List[Tuple[int, int]] = None  # Will default to [(5,5), (7,7), (10,10)]
    num_mazes_per_size: int = 100
    max_steps_per_maze: int = 100
    image_size: int = 300
    
    # Training settings
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_generations: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 512
    max_steps: int = 2000
    save_steps: int = 1000
    
    # Reward weights
    format_reward: float = 1.0
    solve_reward: float = 10.0
    efficiency_bonus: float = 0.1
    partial_credit_weight: float = 5.0
    
    # Curriculum learning settings
    use_curriculum: bool = True
    curriculum_steps_per_level: int = 200  # Steps before introducing next difficulty (increased for better learning)
    curriculum_start_level: int = 0  # Start with easiest (index 0)
    
    def __post_init__(self):
        if self.maze_sizes is None:
            self.maze_sizes = [(5, 5), (6,6), (7, 7), (10, 10)]

