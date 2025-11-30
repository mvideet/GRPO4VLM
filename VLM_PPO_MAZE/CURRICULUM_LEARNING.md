# Curriculum Learning for Maze Environments

This document describes the curriculum learning implementation that progressively increases maze size from small (5x5) to large (100x100) during training.

## Overview

Curriculum learning helps the agent learn progressively by starting with easier tasks (small mazes) and gradually increasing difficulty (larger mazes). This approach has been shown to improve learning efficiency and final performance.

## Features

- **Progressive Size Increase**: Starts at 5x5 and progresses to 100x100
- **Flexible Progression Criteria**: 
  - Success rate-based: Progress when success rate exceeds threshold
  - Update-based: Progress after fixed number of updates
- **Automatic Environment Switching**: Seamlessly transitions between maze sizes
- **Performance Tracking**: Monitors performance at each maze size

## Usage

### Enable Curriculum Learning

Add these flags to your training script:

```bash
--use-curriculum \
--curriculum-start-size 5 \
--curriculum-end-size 100 \
--curriculum-progression success_rate \
--curriculum-success-threshold 0.7 \
--curriculum-min-episodes 100
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-curriculum` | False | Enable curriculum learning |
| `--curriculum-start-size` | 5 | Starting maze size (e.g., 5 for 5x5) |
| `--curriculum-end-size` | 100 | Final maze size (e.g., 100 for 100x100) |
| `--curriculum-progression` | success_rate | Progression criterion: "success_rate" or "updates" |
| `--curriculum-success-threshold` | 0.7 | Success rate needed to progress (0.0-1.0) |
| `--curriculum-min-episodes` | 100 | Minimum episodes before considering progression |
| `--curriculum-updates-per-size` | None | Number of updates per size (if using updates criterion) |

## Progression Criteria

### Success Rate-Based (Recommended)

Progresses to the next maze size when:
1. Minimum number of episodes completed (`--curriculum-min-episodes`)
2. Success rate exceeds threshold (`--curriculum-success-threshold`)

**Example:**
```bash
--curriculum-progression success_rate \
--curriculum-success-threshold 0.7 \
--curriculum-min-episodes 100
```

This means: After at least 100 episodes, if success rate ≥ 70%, progress to next size.

### Update-Based

Progresses after a fixed number of training updates.

**Example:**
```bash
--curriculum-progression updates \
--curriculum-updates-per-size 50
```

This means: Progress to next size after every 50 updates.

## Size Progression

The curriculum automatically creates a geometric progression of maze sizes:

- **Default progression** (5 → 100): 5, 8, 12, 18, 27, 41, 62, 93, 100
- **Custom progression**: You can specify intermediate sizes in code

The progression ensures smooth transitions between sizes.

## How It Works

1. **Initialization**: Starts with smallest maze size (e.g., 5x5)
2. **Training**: Agent trains on current maze size
3. **Monitoring**: Tracks success rate and episode count
4. **Progression Check**: Every 10 updates, checks if progression criteria met
5. **Environment Switch**: If criteria met, creates new environment with larger maze
6. **Continue**: Training continues on new maze size
7. **Repeat**: Process continues until maximum size reached

## Example Training Output

```
=== Curriculum Status ===
Current maze size: 5x5
Progress: 11.1% (1/9)
Episodes: 150, Successes: 120
Success rate: 0.800 (threshold: 0.700)

*** PROGRESSING TO NEXT MAZE SIZE ***
New maze size: 8x8
New environment: maze-sample-8x8-v0
Environment updated successfully!
```

## Visualization Scaling

For larger mazes (≥50x50), the visualization automatically adjusts:
- **Cell size** is reduced to keep image size reasonable (~300px)
- **Image dimensions** scale with maze size
- **Visual elements** (agent, goal, walls) remain clearly visible

## Performance Tracking

The curriculum tracks performance at each maze size:

```python
{
    '5': {'success_rate': 0.85, 'episodes': 200, 'successes': 170},
    '8': {'success_rate': 0.72, 'episodes': 150, 'successes': 108},
    ...
}
```

This information is logged to wandb (if enabled) and printed during training.

## Tips

1. **Start Small**: Begin with 5x5 to establish basic navigation skills
2. **Adjust Threshold**: Lower threshold (0.5-0.6) for faster progression, higher (0.8-0.9) for mastery
3. **Monitor Progress**: Watch curriculum status to understand learning pace
4. **Custom Sizes**: Modify `curriculum.py` to add specific intermediate sizes
5. **Patience**: Larger mazes require more training - be patient!

## Troubleshooting

### Issue: Not progressing to next size
- **Check success rate**: May be below threshold
- **Check episode count**: May not have reached minimum
- **Lower threshold**: Try reducing `--curriculum-success-threshold`

### Issue: Progressing too quickly
- **Increase threshold**: Raise `--curriculum-success-threshold` to 0.8-0.9
- **Increase min episodes**: Set `--curriculum-min-episodes` to 200-500

### Issue: Environment creation fails for large sizes
- **Check gym-maze**: Ensure gym-maze supports custom sizes
- **Fallback**: Code falls back to largest available size and scales visualization

## Integration with Wandb

If wandb is enabled, curriculum metrics are automatically logged:

- `curriculum.current_size`: Current maze size
- `curriculum.progress_percentage`: Overall progress (0-100%)
- `curriculum.current_success_rate`: Success rate on current size
- `curriculum.episode_count`: Episodes on current size
- `curriculum.update_count`: Updates on current size

## Code Structure

- **`a2c_ppo_acktr/curriculum.py`**: Curriculum learning logic
- **`main.py`**: Integration with training loop
- **`a2c_ppo_acktr/envs.py`**: Environment creation with size support
- **`a2c_ppo_acktr/maze_utils.py`**: Visualization scaling for large mazes

## Example Script

See `scripts/run_maze.sh` for a complete example with curriculum learning enabled.

