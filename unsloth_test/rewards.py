# -*- coding: utf-8 -*-
"""
Reward Functions for GRPO Training
"""

import re
from typing import List, Optional
from prompts import ACTIONS_START, ACTIONS_END
from maze_dataset import MazeDatasetGenerator
from maze_env import grpo_act
from config import MazeGRPOConfig
from transformers import TrainerCallback

# Action mapping from text to indices
ACTION_TEXT_TO_IDX = {
    "UP": 0, "NORTH": 0, "N": 0, "U": 0,
    "DOWN": 1, "SOUTH": 1, "S": 1, "D": 1,
    "RIGHT": 2, "EAST": 2, "E": 2, "R": 2,
    "LEFT": 3, "WEST": 3, "W": 3, "L": 3
}


def parse_actions_from_completion(completion: str) -> Optional[List[int]]:
    """Extract action sequence from model completion."""
    pattern = f'{ACTIONS_START}(.*?){ACTIONS_END}'
    match = re.search(pattern, completion, re.DOTALL)
    
    if not match:
        return None
    
    action_text = match.group(1).strip().upper()
    
    # Split by comma, semicolon, newline, or space
    action_strings = re.split(r'[,;\n\s]+', action_text)
    
    actions = []
    for action_str in action_strings:
        action_str = action_str.strip()
        if action_str in ACTION_TEXT_TO_IDX:
            actions.append(ACTION_TEXT_TO_IDX[action_str])
    
    return actions if actions else None


def formatting_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Reward for proper output formatting.
    Uses format_reward from config for each check (reasoning + actions).
    """
    from prompts import REASONING_START, REASONING_END
    
    config = MazeGRPOConfig()  # Get config to use format_reward weight
    reward_per_check = config.format_reward
    
    reasoning_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    actions_pattern = f'{ACTIONS_START}(.*?){ACTIONS_END}'
    
    scores = []
    for completion in completions:
        score = 0.0
        
        reasoning_matches = re.findall(reasoning_pattern, completion, re.DOTALL)
        actions_matches = re.findall(actions_pattern, completion, re.DOTALL)
        
        if len(reasoning_matches) == 1:
            score += reward_per_check
        if len(actions_matches) == 1:
            score += reward_per_check
        
        scores.append(score)
    
    return scores


def maze_execution_reward_func(
    prompts: List[str],
    completions: List[str],
    maze_size: List[List[int]],
    agent_pos: List[List[int]],
    goal_pos: List[List[int]],
    seed: List[int],
    **kwargs
) -> List[float]:
    """
    Execute actions in maze and compute reward based on outcome.
    
    Rewards:
    - Solved: solve_reward + efficiency bonus
    - Partial: dense reward based on distance improvement * partial_credit_weight
    - Invalid actions: -1.0
    
    Also stores execution rewards in a global list for success rate tracking.
    """
    config = MazeGRPOConfig()  # Get config to use reward weights
    scores = []
    generator = MazeDatasetGenerator(config)
    
    # Store execution rewards for success rate tracking (accessible to callbacks)
    if not hasattr(maze_execution_reward_func, '_execution_rewards'):
        maze_execution_reward_func._execution_rewards = []
    
    for i, completion in enumerate(completions):
        # Parse actions from completion
        actions = parse_actions_from_completion(completion)
        
        if actions is None or len(actions) == 0:
            scores.append(-1.0)
            continue
        
        try:
            # Recreate environment with same seed for deterministic maze
            current_seed = seed[i] if isinstance(seed, list) else seed
            current_size = tuple(maze_size[i]) if isinstance(maze_size[i], list) else maze_size[i]
            
            maze_data = generator.generate_single_maze(current_size, current_seed)
            env = maze_data['env']
            
            # Execute action sequence (use config value)
            obs, total_reward, terminated, truncated, info, step_count = grpo_act(
                env, actions, max_steps=config.max_steps_per_maze
            )
            
            # Check if maze was solved
            if terminated:
                # Solved! Give big reward + efficiency bonus
                optimal_steps = abs(goal_pos[i][0] - agent_pos[i][0]) + abs(goal_pos[i][1] - agent_pos[i][1])
                efficiency_bonus = max(0, (optimal_steps * 2 - step_count) * config.efficiency_bonus)
                score = config.solve_reward + efficiency_bonus
            else:
                # Partial credit based on distance improvement (use config weight)
                current_pos = info.get('agent_pos', agent_pos[i])
                goal = goal_pos[i]
                
                initial_dist = abs(agent_pos[i][0] - goal[0]) + abs(agent_pos[i][1] - goal[1])
                final_dist = abs(current_pos[0] - goal[0]) + abs(current_pos[1] - goal[1])
                
                # Reward is proportional to progress made, scaled by partial_credit_weight
                progress = (initial_dist - final_dist) / max(initial_dist, 1)
                score = progress * config.partial_credit_weight
            
            scores.append(score)
            
        except Exception as e:
            print(f"Error executing maze actions: {e}")
            scores.append(-1.0)
    
    # Store execution rewards for success rate computation
    maze_execution_reward_func._execution_rewards.extend(scores)
    
    # Debug: log batch size info (only first time to avoid spam)
    if not hasattr(maze_execution_reward_func, '_logged_batch_info'):
        print(f"[DEBUG] Reward function called with {len(completions)} completions")
        maze_execution_reward_func._logged_batch_info = True
    
    return scores


class SuccessRateCallback(TrainerCallback):
    """
    Callback to compute and log success rate based on execution rewards.
    Success is defined as execution_reward >= 10.0 (maze solved).
    """
    
    def __init__(self, success_threshold: float = 10.0):
        super().__init__()
        self.success_threshold = success_threshold
        self.last_reward_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged. Compute success rate from execution rewards."""
        if logs is None:
            return control
        
        # Get execution rewards from the reward function
        if hasattr(maze_execution_reward_func, '_execution_rewards'):
            execution_rewards = maze_execution_reward_func._execution_rewards
            
            # Only compute if we have new rewards since last log
            if len(execution_rewards) > self.last_reward_count:
                # Get new rewards since last computation
                new_rewards = execution_rewards[self.last_reward_count:]
                self.last_reward_count = len(execution_rewards)
                
                # Compute success rate (execution_reward >= threshold means solved)
                successes = sum(1 for r in new_rewards if r >= self.success_threshold)
                total = len(new_rewards)
                success_rate = successes / total if total > 0 else 0.0
                
                # Log to wandb/metrics
                logs['success_rate'] = success_rate
                logs['success_count'] = successes
                # Note: total_attempts is per gradient accumulation micro-batch
                # With gradient_accumulation_steps=2, this = (batch_size/2) * num_generations = 2 * 4 = 8
                logs['total_attempts'] = total
                
                # Also log cumulative success rate
                cumulative_successes = sum(1 for r in execution_rewards if r >= self.success_threshold)
                cumulative_rate = cumulative_successes / len(execution_rewards) if len(execution_rewards) > 0 else 0.0
                logs['success_rate_cumulative'] = cumulative_rate
            else:
                # No new rewards since last log - log zeros to maintain consistency
                logs['success_rate'] = 0.0
                logs['success_count'] = 0
                logs['total_attempts'] = 0
        
        return control

