# PPO VLM Training for Gym-Maze Environment

## Environment Setup

We suggest installing the python 3.10.0 environment:

```bash
conda create -n vrenv python=3.10.0
conda activate vrenv
cd <path-to-this-repo>
pip install -e ../LLaVA
pip install -e .
pip install gymnasium gym-maze
pip install stable-baselines3 wandb deepspeed sentencepiece git+https://github.com/openai/CLIP.git
pip install xformers
```

Note: Follow the order of installation commands above. LLaVA should be installed first, and xformers should be installed last.

## Install Gym-Maze

```bash
git clone https://github.com/MattChanTK/gym-maze.git
cd gym-maze
pip install -e .
cd ..
```

## Reproduction

```bash
conda activate vrenv
cd scripts
bash run_maze.sh
```

## Code Structure

- `main.py`: Main training script for PPO on maze environment
- `a2c_ppo_acktr/envs.py`: Environment wrappers and vectorization
- `a2c_ppo_acktr/rl_utils.py`: Maze-specific prompts and action parsing
- `a2c_ppo_acktr/maze_utils.py`: Maze visualization and dense reward computation
- `tests/test_maze_components.py`: Unit tests for curriculum, reward, and prompt logic

## Tests

Run the lightweight component tests:

```bash
cd VLM_PPO_MAZE
pytest tests/test_maze_components.py
```

## Smoke Test Training Run

To verify PPO runs end-to-end with your SFT checkpoint (e.g., the ALFWorld SFT model), run a short curriculum-enabled job:

```bash
cd VLM_PPO_MAZE/scripts
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0" accelerate launch \
  --config_file config_zero2.yaml --main_process_port 29501 ../main.py \
  --env-name maze-sample-5x5-v0 \
  --model-path /path/to/your/alfworld_sft_checkpoint \
  --num-env-steps 512 --num-steps 64 --grad-accum-steps 8 \
  --eval-num-per-episode 16 --use-curriculum \
  --curriculum-start-size 5 --curriculum-end-size 20 \
  --curriculum-progression success_rate \
  --curriculum-success-threshold 0.6 \
  --curriculum-min-episodes 20 \
  --use-lora --train-vision all --use-gae
```

This command runs a small number of steps to confirm that PPO, the prompts, and the action parsing all work with your SFT weights.
