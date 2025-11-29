# Implementation Summary: PPO VLM Agent for Gym-Maze Environment

This document summarizes the step-by-step implementation of a PPO-based Visual Language Model (VLM) agent for the gym-maze environment.

## Overview

The implementation adapts the existing PPO training framework from the GRPO4VLM codebase to work with the gym-maze environment. Key changes include:
- Maze environment visualization wrapper
- Dense reward function based on euclidean distance
- Maze-specific prompts and action parsing
- Updated training pipeline

---

## Step 1: Clean Up ALFWorld Code and Set Up Gym-Maze ✅

**Actions Taken:**
- Deleted ALFWorld-specific files:
  - `alf_utils.py`
  - `alf-config.yaml`
  - `alf_conda.yml`
  - `main_alf.py`
- Updated `requirements.txt` to include:
  - `gymnasium`
  - `gym-maze`
  - `pillow` (for image processing)
  - `numpy`
- Updated `README.md` with maze-specific setup instructions

**Files Modified:**
- `requirements.txt`
- `README.md`

---

## Step 2: Create Maze Environment Wrapper with Visualization ✅

**Created:** `a2c_ppo_acktr/maze_utils.py`

**Key Components:**

### `MazeVisualizationWrapper`
- Converts maze state to RGB images for VLM input
- Renders maze with:
  - Green circle: agent position
  - Red circle: goal position
  - Gray cells: visited cells
  - Black walls: obstacles
  - White cells: free space
- Tracks visited cells for visualization
- Handles both single and vectorized environments

### `DenseRewardWrapper`
- Replaces sparse environment rewards with dense euclidean distance rewards
- Computes reward as: `previous_distance - current_distance`
- Provides positive reward for moving closer to goal
- Adds bonus reward when very close to goal

**Key Functions:**
- `_render_maze_image()`: Creates RGB visualization of maze state
- `_extract_agent_position()`: Extracts agent position from various sources
- `_extract_goal_position()`: Extracts goal position from various sources
- `compute_dense_reward()`: Computes reward based on distance reduction

---

## Step 3: Implement Dense Reward Function ✅

**Location:** `a2c_ppo_acktr/maze_utils.py`

**Implementation:**
```python
def compute_dense_reward(previous_pos, current_pos, goal_pos):
    prev_distance = np.linalg.norm(prev_pos - goal)
    curr_distance = np.linalg.norm(curr_pos - goal)
    reward = prev_distance - curr_distance  # Positive when moving closer
    if curr_distance < 0.5:  # Bonus for being very close
        reward += 1.0
    return reward
```

**Features:**
- Rewards progress toward goal (not just reaching it)
- Provides learning signal at every step
- Handles edge cases (None positions, etc.)

---

## Step 4: Update rl_utils.py for Maze Prompts and Action Parsing ✅

**File Modified:** `a2c_ppo_acktr/rl_utils.py`

**Changes:**
1. **Removed ALFWorld code:**
   - Deleted `get_alfworld_prompt()` function
   - Removed ALFWorld imports

2. **Added maze prompts:**
   - Updated `get_prompt()` to handle maze environments
   - Added maze-specific prompt with instructions for navigation
   - Supports both full prompts (with thoughts) and action-only prompts

3. **Updated action parsing:**
   - Added maze actions to `text_projection()`:
     - `["up", "right", "down", "left"]`
     - Maps to gym-maze action space: up=0, right=1, down=2, left=3

**Maze Prompt Example:**
```
"You are navigating a maze. You can see the maze layout in the image.
The green circle represents your current position, and the red circle 
represents the goal. Your goal is to navigate from your current position 
to the goal by choosing the correct direction. You can choose between 
the following actions: ['up', 'down', 'left', 'right']."
```

---

## Step 5: Update envs.py to Support Maze Environment ✅

**File Modified:** `a2c_ppo_acktr/envs.py`

**Changes:**
1. **Updated `make_env()` function:**
   - Added detection for maze environments
   - Applies `DenseRewardWrapper` first
   - Then applies `MazeVisualizationWrapper`
   - Ensures proper wrapper order

2. **Updated `make_vec_envs()` function:**
   - Skips observation normalization for maze (already images)
   - Skips frame stacking for maze (single frame sufficient)
   - Maintains compatibility with other environments

**Wrapper Order:**
```
Base Maze Env → DenseRewardWrapper → MazeVisualizationWrapper → VecEnv
```

---

## Step 6: Create main.py for Maze PPO Training ✅

**Created:** `main.py`

**Key Features:**
- Based on `VLM_PPO/main.py` but adapted for maze
- Supports maze environment detection
- Uses maze-specific prompts and action parsing
- Maintains all PPO training logic
- Supports LoRA fine-tuning
- Compatible with DeepSpeed

**Environment Detection:**
```python
if "maze" in args.env_name.lower() or "gym_maze" in args.env_name.lower():
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, None, device, False, 1)
```

---

## Step 7: Update Scripts, Requirements, and README ✅

**Files Created/Modified:**

1. **`scripts/run_maze.sh`:**
   - Training script for maze environment
   - Uses `maze-sample-5x5-v0` by default
   - Configurable via command-line arguments

2. **`scripts/config_zero2.yaml`:**
   - DeepSpeed configuration
   - Set `num_processes: 1` (can be increased for multi-GPU)

3. **`README.md`:**
   - Updated with maze-specific instructions
   - Installation steps for gym-maze
   - Usage examples

---

## File Structure

```
VLM_PPO_MAZE/
├── a2c_ppo_acktr/
│   ├── maze_utils.py          # NEW: Visualization and dense reward
│   ├── rl_utils.py            # MODIFIED: Maze prompts and actions
│   ├── envs.py                # MODIFIED: Maze environment support
│   ├── model.py                # (unchanged)
│   ├── storage.py              # (unchanged)
│   └── ...
├── main.py                     # NEW: Maze training script
├── scripts/
│   ├── run_maze.sh            # NEW: Training script
│   └── config_zero2.yaml     # (updated)
├── requirements.txt           # MODIFIED: Added gym-maze
└── README.md                  # MODIFIED: Maze instructions
```

---

## Usage

### 1. Install Dependencies
```bash
conda create -n vrenv python=3.10.0
conda activate vrenv
cd VLM_PPO_MAZE
pip install -e ../LLaVA
pip install -e .
pip install gymnasium gym-maze
pip install stable-baselines3 wandb deepspeed sentencepiece git+https://github.com/openai/CLIP.git
pip install xformers
```

### 2. Install Gym-Maze
```bash
git clone https://github.com/MattChanTK/gym-maze.git
cd gym-maze
pip install -e .
cd ..
```

### 3. Run Training
```bash
cd scripts
bash run_maze.sh
```

### 4. Customize Training
Edit `scripts/run_maze.sh` to:
- Change environment: `--env-name maze-sample-10x10-v0`
- Adjust learning rate: `--init-lr 1e-5`
- Set model path: `--model-path /path/to/model`
- Enable wandb: uncomment wandb lines

---

## Key Features

### Dense Reward Function
- **Formula:** `reward = previous_distance - current_distance`
- **Benefits:**
  - Provides learning signal at every step
  - Encourages progress toward goal
  - Faster convergence than sparse rewards

### Visualization
- **RGB Images:** 300x300x3 (configurable via `cell_size`)
- **Visual Elements:**
  - Agent: Green circle
  - Goal: Red circle
  - Visited: Gray cells
  - Walls: Black
  - Free space: White

### Action Space
- **Actions:** `["up", "right", "down", "left"]`
- **Mapping:** up=0, right=1, down=2, left=3
- **JSON Format:** `{"action": "up"}` or `{"thoughts": "...", "action": "up"}`

---

## Testing

To test the implementation:

1. **Test Environment:**
```python
import gymnasium as gym
import gym_maze
from a2c_ppo_acktr.maze_utils import MazeVisualizationWrapper, DenseRewardWrapper

env = gym.make('maze-sample-5x5-v0')
env = DenseRewardWrapper(env)
env = MazeVisualizationWrapper(env)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # Should be (height, width, 3)
print(f"Agent pos: {info.get('agent_pos')}")
print(f"Goal pos: {info.get('goal_pos')}")
```

2. **Test Reward:**
```python
obs, reward, done, truncated, info = env.step(0)  # Move up
print(f"Reward: {reward}")  # Should be positive if moving toward goal
```

---

## Next Steps

1. **Test with different maze sizes:**
   - `maze-sample-5x5-v0`
   - `maze-sample-10x10-v0`
   - `maze-random-10x10-v0`

2. **Tune hyperparameters:**
   - Learning rate
   - PPO epochs
   - Gradient accumulation steps

3. **Experiment with reward shaping:**
   - Adjust bonus reward threshold
   - Add penalty for revisiting cells
   - Scale reward magnitude

4. **Monitor training:**
   - Enable wandb logging
   - Track success rate
   - Monitor reward distribution

---

## Troubleshooting

### Issue: Environment not found
**Solution:** Ensure gym-maze is installed:
```bash
pip install -e /path/to/gym-maze
```

### Issue: Position extraction fails
**Solution:** Check that environment provides position in `info` dict or observation. May need to adjust `_extract_agent_position()` based on your gym-maze version.

### Issue: Visualization not working
**Solution:** Check that `maze_size` is correctly inferred. You may need to manually set it in `MazeVisualizationWrapper.__init__()`.

### Issue: Reward always zero
**Solution:** Verify that positions are being extracted correctly. Add debug prints in `compute_dense_reward()`.

---

## Notes

- The implementation assumes gym-maze returns positions in `info` dict or observation
- Maze structure extraction may need adjustment based on gym-maze version
- Visualization cell size can be adjusted for different maze sizes
- Dense reward can be further tuned for better learning

---

## Citation

If you use this implementation, please cite the original paper:
```bibtex
@inproceedings{
zhai2024finetuning,
title={Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning},
author={Yuexiang Zhai and Hao Bai and Zipeng Lin and Jiayi Pan and Shengbang Tong and Yifei Zhou and Alane Suhr and Saining Xie and Yann LeCun and Yi Ma and Sergey Levine},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=nBjmMF2IZU}
}
```


