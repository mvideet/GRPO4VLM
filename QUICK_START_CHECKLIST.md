# Quick Start Checklist: Blackjack PPO Training

Use this checklist to quickly set up and run the training.

## Pre-flight Checklist

- [ ] Python 3.10.0 installed
- [ ] CUDA-capable GPU(s) available
- [ ] Sufficient disk space for models and checkpoints
- [ ] Internet connection for downloading models

## Setup Steps

### 1. Environment Setup
```bash
conda create -n vrenv python=3.10.0
conda activate vrenv
cd /Users/shauryaagrawal/GRPO4VLM
```

### 2. Install Dependencies (IN ORDER!)
```bash
pip install -e ./LLaVA
pip install -e ./gym-cards
pip install gymnasium[atari,accept-rom-license]
pip install stable-baselines3 wandb deepspeed sentencepiece git+https://github.com/openai/CLIP.git
pip install xformers
```

### 3. Verify Installation
```bash
python -c "import gym_cards; import gymnasium as gym; env = gym.make('gym_cards/Blackjack-v0'); print('OK')"
```

### 4. Configure Training

#### Edit `VLM_PPO/scripts/config_zero2.yaml`:
- [ ] Set `num_processes: X` (X = number of GPUs you have)

#### Edit `VLM_PPO/scripts/run_bj.sh`:
- [ ] Set `CUDA_VISIBLE_DEVICES="0,1,..."` (match your GPU IDs)
- [ ] Set `--model-path` to either:
  - Your SFT checkpoint path, OR
  - `liuhaotian/llava-v1.6-mistral-7b` (pre-trained model)
- [ ] (Optional) Uncomment wandb lines if you want logging

### 5. Run Training
```bash
cd VLM_PPO/scripts
bash run_bj.sh
```

## Key Parameters to Know

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_processes` | 2 | Number of GPUs/processes |
| `--num-env-steps` | 15000 | Total training steps |
| `--num-steps` | 512 | Steps per PPO update |
| `--ppo-epoch` | 4 | PPO epochs per update |
| `--init-lr` | 1e-5 | Initial learning rate |
| `--use-lora` | ✓ | Use LoRA for efficiency |

## Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Out of memory | Reduce `num_processes` to 1 |
| Environment not found | `pip install -e ./gym-cards` |
| Model download fails | Check internet/HuggingFace access |
| NCCL timeout | Use single GPU (`num_processes: 1`) |

## Expected Output

You should see:
- Episode rewards (mean/median)
- Success rate
- Training losses (value, action, entropy)
- FPS (frames per second)

## Next Steps After Training

- [ ] Check saved checkpoints
- [ ] Evaluate model performance
- [ ] Compare with paper results
- [ ] Experiment with hyperparameters

## Need Help?

See `TRAINING_GUIDE.md` for detailed explanations and troubleshooting.


