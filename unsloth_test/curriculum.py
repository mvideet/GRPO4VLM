# -*- coding: utf-8 -*-
"""
Curriculum Learning for Maze GRPO Training
"""

from datasets import Dataset
from transformers import TrainerCallback
from config import MazeGRPOConfig


class CurriculumLearningCallback:
    """Callback for curriculum learning - gradually increases maze difficulty."""
    
    def __init__(self, config: MazeGRPOConfig, full_dataset: Dataset):
        self.config = config
        self.full_dataset = full_dataset
        self.current_level = config.curriculum_start_level
        self.steps_per_level = config.curriculum_steps_per_level
        self.max_level = len(config.maze_sizes) - 1
        self.last_level_change_step = 0
        
    def get_current_dataset(self, current_step: int) -> Dataset:
        """Get filtered dataset based on current curriculum level."""
        if not self.config.use_curriculum:
            return self.full_dataset
        
        # Check if we should advance to next level
        steps_since_last_change = current_step - self.last_level_change_step
        if (self.current_level < self.max_level and 
            steps_since_last_change >= self.steps_per_level):
            self.current_level += 1
            self.last_level_change_step = current_step
            print(f"\n🎓 Curriculum: Advancing to level {self.current_level} "
                  f"({self.config.maze_sizes[self.current_level]}) at step {current_step}")
        
        # Filter dataset to include only mazes up to current difficulty level
        def filter_by_level(example):
            return example.get('difficulty_level', 0) <= self.current_level
        
        filtered_dataset = self.full_dataset.filter(filter_by_level)
        
        return filtered_dataset
    
    def get_current_level_info(self) -> dict:
        """Get information about current curriculum level."""
        if not self.config.use_curriculum:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "current_level": self.current_level,
            "current_maze_size": self.config.maze_sizes[self.current_level],
            "max_level": self.max_level,
            "total_levels": len(self.config.maze_sizes)
        }


class CurriculumTrainerCallback(TrainerCallback):
    """Trainer callback that implements curriculum learning by updating the dataset."""
    
    def __init__(self, curriculum_callback: CurriculumLearningCallback):
        super().__init__()
        self.curriculum_callback = curriculum_callback
        self.last_level = curriculum_callback.current_level
        self.trainer = None
    
    def set_trainer(self, trainer):
        """Set the trainer reference."""
        self.trainer = trainer
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training begins."""
        # Try to get trainer from kwargs if not already set
        if self.trainer is None:
            self.trainer = kwargs.get('trainer', None)
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update dataset when curriculum level changes."""
        if not self.curriculum_callback.config.use_curriculum:
            return control
        
        current_step = state.global_step
        
        # Check if we need to update the dataset
        steps_since_last_change = current_step - self.curriculum_callback.last_level_change_step
        if (self.curriculum_callback.current_level < self.curriculum_callback.max_level and 
            steps_since_last_change >= self.curriculum_callback.steps_per_level):
            
            # Advance curriculum level
            self.curriculum_callback.current_level += 1
            self.curriculum_callback.last_level_change_step = current_step
            
            level_info = self.curriculum_callback.get_current_level_info()
            print(f"\n🎓 Curriculum Learning: Advanced to Level {level_info['current_level']} "
                  f"(Maze size: {level_info['current_maze_size']}) at step {current_step}")
            
            # Log to wandb if available
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "curriculum/level": level_info['current_level'],
                        "curriculum/maze_size": f"{level_info['current_maze_size'][0]}x{level_info['current_maze_size'][1]}",
                    }, step=current_step)
            except:
                pass
        
        return control

