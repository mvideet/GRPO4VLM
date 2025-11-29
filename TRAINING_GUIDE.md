# Step-by-Step Guide: Recreating PPO Training for Blackjack Environment

This guide will walk you through the complete process of recreating the PPO training implementation from the paper "Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning" for the gym-cards/Blackjack environment.

## Overview

The training pipeline consists of two main steps:
1. **Supervised Fine-Tuning (SFT)**: Prepare an initial checkpoint with instruction-following data
2. **PPO Training**: Fine-tune the VLM using Proximal Policy Optimization on the Blackjack environment

---

## Prerequisites

- Python 3.10.0
- CUDA-capable GPUs (recommended: 2+ GPUs for multi-GPU training)
- Sufficient GPU memory (at least 16GB per GPU recommended)
- Access to Hugging Face models

---

## Step 1: Environment Setup

### 1.1 Create Conda Environment

```bash
conda create -n vrenv python=3.10.0
conda activate vrenv
```

### 1.2 Install Dependencies

Navigate to the repository root and install packages in the following order:

```bash
cd /Users/shauryaagrawal/GRPO4VLM

# Install LLaVA first
pip install -e ./LLaVA

# Install gym-cards environment
pip install -e ./gym-cards

# Install other dependencies
pip install gymnasium[atari,accept-rom-license]
pip install stable-baselines3 wandb deepspeed sentencepiece git+https://github.com/openai/CLIP.git

# Install xformers last (important!)
pip install xformers
```

**Note**: The order matters! LLaVA must be installed first, and xformers should be installed last.

### 1.3 Verify Installation

Test that the environment is set up correctly:

```bash
python -c "import gym_cards; import gymnasium as gym; env = gym.make('gym_cards/Blackjack-v0'); print('Environment loaded successfully!')"
```

---

## Step 2: Supervised Fine-Tuning (SFT) - Optional but Recommended

### 2.1 Download SFT Data

The paper uses instruction-following data for initial SFT. You can:
- Download the data from: https://huggingface.co/LEVI-Project/sft-data/tree/main
- Or skip this step and use the pre-trained model directly (see Step 2.3)

### 2.2 Run SFT Training

If you have SFT data, modify `finetune.sh` with your paths:

```bash
# Edit finetune.sh and update:
# - --data_path: path to your JSON training data
# - --image_folder: path to your image folder
# - --output_dir: where to save the SFT checkpoint

# Then run:
cd /Users/shauryaagrawal/GRPO4VLM
bash finetune.sh
```

**Important parameters in finetune.sh:**
- `--model_name_or_path`: Base model (default: `liuhaotian/llava-v1.6-mistral-7b`)
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--output_dir`: Where to save your SFT checkpoint

### 2.3 Alternative: Use Pre-trained Model

If you want to skip SFT and start directly with PPO, you can use the pre-trained model:
- Model path: `liuhaotian/llava-v1.6-mistral-7b`

This will be used in the PPO training step.

---

## Step 3: Configure PPO Training for Blackjack

### 3.1 Update Configuration Files

#### Edit `VLM_PPO/scripts/config_zero2.yaml`:

```yaml
num_processes: 2  # Set this to the number of GPUs you have
```

**Important**: Make sure `num_processes` matches the number of GPUs you'll use.

#### Edit `VLM_PPO/scripts/run_bj.sh`:

Update the following key parameters:

```bash
# Set GPU IDs (must be >= num_processes in config_zero2.yaml)
CUDA_VISIBLE_DEVICES="0,1"  # Use your available GPU IDs

# Set the model path
--model-path /your_sft_checkpoint_for_blackjack
# OR if using pre-trained model directly:
--model-path liuhaotian/llava-v1.6-mistral-7b

# Optional: Enable wandb logging
--wandb-project your_wandb_proj
--wandb-run your_wandb_run
--use-wandb
```

### 3.2 Key Training Parameters Explained

From `run_bj.sh`, here are the important parameters:

- `--env-name gym_cards/Blackjack-v0`: The environment to train on
- `--init-lr 1e-5`: Initial learning rate for cosine scheduler
- `--end-lr 1e-9`: Final learning rate for cosine scheduler
- `--lr_max_steps 25`: Number of steps for cosine annealing
- `--eval-num-per-episode 1000`: Episodes to evaluate performance
- `--num-env-steps 15000`: Total environment steps for training
- `--num-steps 512`: Environment steps collected per PPO update
- `--grad-accum-steps 128`: Gradient accumulation steps
- `--max-new-tokens 256`: Maximum tokens for text action generation
- `--thought-prob-coef 0.5`: Scaling factor for Chain-of-Thought tokens
- `--use-gae`: Use Generalized Advantage Estimation
- `--seed 1`: Random seed for reproducibility
- `--temperature 0.2`: Sampling temperature
- `--ppo-epoch 4`: Number of PPO epochs per update
- `--mini-batch-size 1`: Mini-batch size for PPO updates
- `--use-lora`: Use LoRA for efficient fine-tuning
- `--train-vision all`: Which vision components to train

---

## Step 4: Run PPO Training

### 4.1 Navigate to Scripts Directory

```bash
cd /Users/shauryaagrawal/GRPO4VLM/VLM_PPO/scripts
```

### 4.2 Start Training

```bash
bash run_bj.sh
```

### 4.3 Monitor Training

The training will output:
- Episode rewards (mean, median, min, max)
- Success rate
- Value loss, action loss, entropy
- FPS (frames per second)

If you enabled wandb, you can also monitor training in the Weights & Biases dashboard.

---

## Step 5: Understanding the Training Process

### 5.1 How PPO Works in This Codebase

1. **Environment Interaction**: The VLM observes pixel images from the Blackjack game
2. **Action Generation**: The model generates text actions (JSON format with "stand" or "hit")
3. **Action Parsing**: Text actions are parsed into discrete actions (0=stand, 1=hit)
4. **Reward Collection**: Rewards are collected from the environment (+1 win, -1 lose, 0 draw)
5. **PPO Update**: The model is updated using PPO algorithm with GAE
6. **Evaluation**: Performance is evaluated periodically

### 5.2 Blackjack Environment Details

- **Action Space**: Discrete(2) - 0: Stand, 1: Hit
- **Observation Space**: Box(300, 300, 3) - RGB pixel images
- **Rewards**: 
  - +1 for winning
  - -1 for losing
  - 0 for draw
  - +1.5 for natural blackjack (if natural=True)

### 5.3 Model Architecture

- **Base Model**: LLaVA-1.6-Mistral-7B (vision-language model)
- **Training Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Components Trained**: Vision encoder + language model (configurable)

---

## Step 6: Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `num_processes` in `config_zero2.yaml`
   - Reduce `--num-steps` or `--grad-accum-steps`
   - Enable quantization: add `--q4` or `--q8` flag (note: doesn't work with LoRA)

2. **NCCL Timeout (Multi-GPU)**
   - Reduce `num_processes` to 1 if using single GPU
   - Increase timeout in deepspeed config

3. **Token ID Mismatch**
   - The codebase has hardcoded token IDs that may be outdated
   - Check `VLM_PPO/a2c_ppo_acktr/llava_interface/interface.py` line 66
   - Manually verify token IDs for `"action":` string

4. **Environment Not Found**
   - Ensure gym-cards is installed: `pip install -e ./gym-cards`
   - Verify installation: `python -c "import gym_cards"`

5. **Model Loading Issues**
   - Ensure you have internet connection for downloading models
   - Check Hugging Face access if using private models
   - Verify model path is correct

### Debugging Tips

- Start with single GPU (`num_processes=1`) to debug
- Use smaller `--num-env-steps` for quick tests
- Check logs for specific error messages
- Verify all dependencies are installed correctly

---

## Step 7: Expected Results

Based on the paper, you should see:
- **Learning Progress**: Episode rewards should increase over time
- **Success Rate**: Should improve as training progresses
- **Training Time**: Approximately several hours depending on hardware

The model should learn to:
- Observe the game state from pixel images
- Make strategic decisions (stand/hit) based on current hand and dealer's visible card
- Generate valid JSON-formatted actions

---

## Step 8: Saving and Loading Checkpoints

The training process will save checkpoints. To resume training:
- Modify `--model-path` to point to your saved checkpoint
- The training will continue from the checkpoint

---

## Additional Resources

- **Paper**: https://arxiv.org/abs/2405.10292
- **Project Page**: https://rl4vlm.github.io/
- **LLaVA Repository**: https://github.com/haotian-liu/LLaVA
- **GymCards Documentation**: See `gym-cards/README.md`

---

## Quick Start (Minimal Setup)

If you want to quickly test the setup without full SFT:

```bash
# 1. Setup environment
conda create -n vrenv python=3.10.0
conda activate vrenv
cd /Users/shauryaagrawal/GRPO4VLM
pip install -e ./LLaVA
pip install -e ./gym-cards
pip install gymnasium[atari,accept-rom-license] stable-baselines3 wandb deepspeed sentencepiece git+https://github.com/openai/CLIP.git xformers

# 2. Edit run_bj.sh to use pre-trained model
# Change: --model-path liuhaotian/llava-v1.6-mistral-7b

# 3. Update config_zero2.yaml: num_processes: 1 (for single GPU)

# 4. Run training
cd VLM_PPO/scripts
bash run_bj.sh
```

---

## Citation

If you use this codebase, please cite the original paper:

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


