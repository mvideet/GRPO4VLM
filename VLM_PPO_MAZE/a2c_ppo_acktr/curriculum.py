from typing import List, Optional


class MazeCurriculum:   
    def __init__(
        self,
        start_size: int = 5,
        end_size: int = 100,
        size_steps: Optional[List[int]] = None,
        progression_criterion: str = "success_rate",
        success_rate_threshold: float = 0.7,
        min_episodes_per_size: int = 100,
        updates_per_size: Optional[int] = None,
    ):
        self.start_size = start_size
        self.end_size = end_size
        
        if size_steps is None:
            base_schedule = [5, 10, 100]
        else:
            base_schedule = size_steps

        filtered = [s for s in base_schedule if start_size <= s <= end_size]
        if not filtered:
            filtered = [start_size, min(end_size, 100)]
        self.size_steps = sorted(set(filtered))
        
        self.progression_criterion = progression_criterion
        self.success_rate_threshold = success_rate_threshold
        self.min_episodes_per_size = min_episodes_per_size
        self.updates_per_size = updates_per_size
    
        self.current_size_idx = 0
        self.current_size = self.size_steps[0]
        self.episode_count = 0
        self.success_count = 0
        self.update_count = 0
        self.size_performance = {}
                 
    def get_current_size(self) -> int:
        return self.current_size
    
    def get_current_env_name(self) -> str:
        size = self.current_size
        return f"maze-random-{size}x{size}-v0"
    
    def record_episode(self, success: bool):
        self.episode_count += 1
        if success:
            self.success_count += 1
    
    def record_update(self):
        self.update_count += 1
    
    def should_progress(self) -> bool:
        if self.current_size_idx >= len(self.size_steps) - 1:
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
        if not self.should_progress():
            return False
        
        if self.current_size_idx >= len(self.size_steps) - 1:
            return False
    
        if self.episode_count > 0:
            success_rate = self.success_count / self.episode_count
            self.size_performance[self.current_size] = {
                'success_rate': success_rate,
                'episodes': self.episode_count,
                'successes': self.success_count
            }
        
        self.current_size_idx += 1
        self.current_size = self.size_steps[self.current_size_idx]
        self.episode_count = 0
        self.success_count = 0
        self.update_count = 0
        
        return True
    
    def get_progress_info(self) -> dict:
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
        self.episode_count = 0
        self.success_count = 0
        self.update_count = 0

