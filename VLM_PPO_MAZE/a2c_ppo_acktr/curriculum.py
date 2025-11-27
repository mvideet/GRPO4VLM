"""
Curriculum learning for maze environments.
Progressively increases maze size from small to large.
"""
import numpy as np
from typing import List, Tuple, Optional


class MazeCurriculum:
    """
    Manages curriculum learning for maze environments.
    Progressively increases maze size based on performance.
    """
    
    def __init__(
        self,
        start_size: int = 5,
        end_size: int = 100,
        size_steps: Optional[List[int]] = None,
        progression_criterion: str = "success_rate",  # "success_rate" or "updates"
        success_rate_threshold: float = 0.7,
        min_episodes_per_size: int = 100,
        updates_per_size: Optional[int] = None,
    ):
        """
        Initialize curriculum learning.
        
        Args:
            start_size: Starting maze size (e.g., 5 for 5x5)
            end_size: Final maze size (e.g., 100 for 100x100)
            size_steps: List of intermediate sizes. If None, creates geometric progression
            progression_criterion: "success_rate" or "updates"
            success_rate_threshold: Success rate needed to progress (if using success_rate)
            min_episodes_per_size: Minimum episodes before considering progression
            updates_per_size: Number of updates per size (if using updates criterion)
        """
        self.start_size = start_size
        self.end_size = end_size
        
        # Create size progression
        if size_steps is None:
            # Create geometric progression from start to end
            sizes = [start_size]
            current = start_size
            while current < end_size:
                # Increase by ~1.5x each step, but ensure we reach end_size
                next_size = min(int(current * 1.5), end_size)
                if next_size > current:
                    sizes.append(next_size)
                    current = next_size
                else:
                    sizes.append(end_size)
                    break
            if sizes[-1] != end_size:
                sizes.append(end_size)
            self.size_steps = sorted(list(set(sizes)))
        else:
            self.size_steps = sorted(list(set([start_size] + size_steps + [end_size])))
        
        self.progression_criterion = progression_criterion
        self.success_rate_threshold = success_rate_threshold
        self.min_episodes_per_size = min_episodes_per_size
        self.updates_per_size = updates_per_size
        
        # Current curriculum state
        self.current_size_idx = 0
        self.current_size = self.size_steps[0]
        self.episode_count = 0
        self.success_count = 0
        self.update_count = 0
        
        # Track performance per size
        self.size_performance = {}
        
    def get_current_size(self) -> int:
        """Get current maze size."""
        return self.current_size
    
    def get_current_env_name(self) -> str:
        """Get current environment name for gym-maze."""
        size = self.current_size
        # For standard sizes, use predefined environments
        if size <= 10:
            return f"maze-sample-{size}x{size}-v0"
        else:
            # For larger sizes, we'll need to create custom mazes
            # For now, use a pattern that can be handled by make_env
            return f"maze-custom-{size}x{size}-v0"
    
    def record_episode(self, success: bool):
        """Record an episode result."""
        self.episode_count += 1
        if success:
            self.success_count += 1
    
    def record_update(self):
        """Record a training update."""
        self.update_count += 1
    
    def should_progress(self) -> bool:
        """
        Check if we should progress to the next maze size.
        
        Returns:
            True if should progress, False otherwise
        """
        if self.current_size_idx >= len(self.size_steps) - 1:
            # Already at maximum size
            return False
        
        if self.progression_criterion == "success_rate":
            if self.episode_count < self.min_episodes_per_size:
                return False
            success_rate = self.success_count / self.episode_count
            return success_rate >= self.success_rate_threshold
        
        elif self.progression_criterion == "updates":
            if self.updates_per_size is None:
                raise ValueError("updates_per_size must be set when using 'updates' criterion")
            return self.update_count >= self.updates_per_size
        
        else:
            raise ValueError(f"Unknown progression criterion: {self.progression_criterion}")
    
    def progress(self) -> bool:
        """
        Progress to the next maze size.
        
        Returns:
            True if progressed, False if already at max size
        """
        if not self.should_progress():
            return False
        
        if self.current_size_idx >= len(self.size_steps) - 1:
            return False
        
        # Save performance for current size
        if self.episode_count > 0:
            success_rate = self.success_count / self.episode_count
            self.size_performance[self.current_size] = {
                'success_rate': success_rate,
                'episodes': self.episode_count,
                'successes': self.success_count
            }
        
        # Move to next size
        self.current_size_idx += 1
        self.current_size = self.size_steps[self.current_size_idx]
        
        # Reset counters
        self.episode_count = 0
        self.success_count = 0
        self.update_count = 0
        
        return True
    
    def get_progress_info(self) -> dict:
        """Get information about curriculum progress."""
        total_sizes = len(self.size_steps)
        progress_pct = (self.current_size_idx + 1) / total_sizes * 100
        
        info = {
            'current_size': self.current_size,
            'current_size_idx': self.current_size_idx,
            'total_sizes': total_sizes,
            'progress_percentage': progress_pct,
            'episode_count': self.episode_count,
            'success_count': self.success_count,
            'update_count': self.update_count,
        }
        
        if self.episode_count > 0:
            info['current_success_rate'] = self.success_count / self.episode_count
        else:
            info['current_success_rate'] = 0.0
        
        if self.progression_criterion == "success_rate":
            info['threshold'] = self.success_rate_threshold
            info['meets_threshold'] = info['current_success_rate'] >= self.success_rate_threshold
        elif self.progression_criterion == "updates":
            info['threshold'] = self.updates_per_size
            info['meets_threshold'] = self.update_count >= self.updates_per_size
        
        info['size_performance'] = self.size_performance.copy()
        
        return info
    
    def reset_counters(self):
        """Reset episode and update counters (useful when switching environments)."""
        self.episode_count = 0
        self.success_count = 0
        self.update_count = 0


def create_custom_maze_env(size: int, seed: Optional[int] = None):
    """
    Create a custom maze environment of specified size.
    
    For sizes > 10, gym-maze may not have predefined environments,
    so we use the largest available and scale visualization.
    
    Args:
        size: Maze size (e.g., 20 for 20x20)
        seed: Random seed
    
    Returns:
        gym.Env: Maze environment
    """
    import gymnasium as gym
    import gym_maze
    
    # Try to create environment with custom size
    # Note: gym-maze may not support all sizes directly
    # We'll use the largest available size and handle scaling in visualization
    
    try:
        # Try standard sizes first
        if size <= 10:
            env_id = f"maze-sample-{size}x{size}-v0"
            try:
                env = gym.make(env_id)
                if seed is not None:
                    env.reset(seed=seed)
                return env
            except:
                pass
        
        # For larger sizes, try random mazes if available
        if size <= 50:
            env_id = f"maze-random-{size}x{size}-v0"
            try:
                env = gym.make(env_id)
                if seed is not None:
                    env.reset(seed=seed)
                return env
            except:
                pass
        
        # Fallback: use largest available size (10x10 or 50x50)
        # Visualization wrapper will handle scaling
        fallback_sizes = [50, 20, 10, 5]
        for fallback_size in fallback_sizes:
            try:
                env_id = f"maze-sample-{fallback_size}x{fallback_size}-v0"
                env = gym.make(env_id)
                if seed is not None:
                    env.reset(seed=seed)
                print(f"Note: Using {fallback_size}x{fallback_size} maze as base for {size}x{size} visualization")
                return env
            except:
                continue
        
        # Last resort: use any available maze
        env = gym.make("maze-sample-5x5-v0")
        if seed is not None:
            env.reset(seed=seed)
        print(f"Warning: Using 5x5 maze as fallback for {size}x{size} (visualization will be scaled)")
        return env
        
    except Exception as e:
        # Final fallback
        print(f"Warning: Could not create {size}x{size} maze: {e}")
        print(f"Using 5x5 maze as fallback (visualization will be scaled)")
        env = gym.make("maze-sample-5x5-v0")
        if seed is not None:
            env.reset(seed=seed)
        return env

