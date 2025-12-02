from patch import replace_llama_attn_with_xformers_attn, XFORMERS_AVAILABLE
replace_llama_attn_with_xformers_attn()
if XFORMERS_AVAILABLE:
    print("using xformers")
else:
    print("using standard PyTorch attention (xformers not available)")

import copy
import glob
import os
import time
from collections import deque

import gym
import gym_maze
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.algo.grpo import GRPO
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, text_projection_multi_actions
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.grpo_model import GRPOVLMPolicy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.grpo_storage import GRPORolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate, grpo_llava_generate, grpo_llava_evaluate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model
from a2c_ppo_acktr.curriculum import MazeCurriculum
from a2c_ppo_acktr.maze_utils import grpo_act

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

import math
import random
from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor
import transformers
from PIL import Image

from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

import warnings
warnings.filterwarnings("ignore")

def main():
    args = get_args()
    
    # Debug: Print max_new_tokens to verify it's being set correctly
    print(f"DEBUG: max_new_tokens = {args.max_new_tokens}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    ## environment interaction device is cpu
    model_device = device

    #initialization of llava
    model_path = args.model_path
    cache_dir = args.cache_dir

    print(model_path)
    #load_pretrained_model(model_path, model_path, model_path)
    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
        if args.q8 or args.q4:
            raise ValueError("Lora model does not support 8bit or 4bit quantization")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            print("8bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )
            print("4bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    use_grad_ckpt = True
    if use_grad_ckpt:
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    base.config.max_length = 1024
    print("Model max context length:{}".format(base.config.max_length))
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor

    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)
    # For GRPO, we don't need VLMValue (no value function)
    # value_model = VLMValue(base)
    # value_model = value_model.to(model_device)

    # Curriculum learning disabled - using fixed 5x5 mazes only
    curriculum = None
    current_env_name = args.env_name
    current_maze_size = 5  # Fixed 5x5 maze size

    # Create maze environment - fixed 5x5 mazes
    if "maze" in args.env_name.lower() or "gym_maze" in args.env_name.lower():
        envs = make_vec_envs(current_env_name, args.seed, args.num_processes,
                             args.gamma, None, device, False, 1, maze_size=current_maze_size)
    else:
        print("Environment not supported. Please use a maze environment (e.g., 'maze-sample-5x5-v0')")
        exit(1)


    obs = envs.reset()
    infos = None
    ## Inputing Prompt here
    # Use current environment name (may change with curriculum)
    prompt_env_name = current_env_name if curriculum is not None else args.env_name
    qs = get_prompt(prompt_env_name, args.action_only_prompt, infos)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

    # Use multi-action projection for GRPO (parses full action sequences)
    projection_f = partial(text_projection_multi_actions, env_name=prompt_env_name)
    
    # Get num_samples from args (number of generations per observation)
    num_samples = getattr(args, 'num_generations', 4)  # Default to 4 generations

    actor_critic = GRPOVLMPolicy(tokenizer=tokenizer,
                                 image_processor=image_processor,
                                 base=base,
                                 projection_f=projection_f,
                                 INPUT_IDS=INPUT_IDS,
                                 args=args,
                                 num_samples=num_samples)
    # For GRPO, optimize the base model parameters
    optimizer = optim.Adam(actor_critic.base.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    # Configure DeepSpeed if enabled, otherwise skip
    if hasattr(AcceleratorState(), 'deepspeed_plugin') and AcceleratorState().deepspeed_plugin is not None:
        AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    # Initialize GRPO agent (uses existing GRPO class from algo/grpo.py)
    agent = GRPO(
        policy=actor_critic,
        reference_model=None,  # No reference model (beta=0 means no KL penalty)
        optimizer=optimizer,
        accelerator=accelerator,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        mini_batch_size=args.mini_batch_size,
        value_loss_coef=args.value_loss_coef,  # Not used but required by GRPO class
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        num_generations=num_samples,
        beta=0.0  # No KL penalty (no reference model)
    )
    
    rollouts = GRPORolloutStorage(args.num_processes,
                                  envs.observation_space.shape,
                                  envs.action_space,
                                  args.max_new_tokens,
                                  num_generations=num_samples)

    # Test GRPO act (only print on first iteration)
    output_ids_list, action_list, action_tokens_log_prob_list = actor_critic.act(obs, INPUT_IDS=INPUT_IDS)
    print("Number of generations per process: {}".format(len(action_list)))
    print("First generation actions: {}".format(action_list[0] if action_list else None))
    print("Action tokens log prob shape: {}".format(action_tokens_log_prob_list[0].shape if action_tokens_log_prob_list else None))

    rollouts.to(device)

    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)

    start = time.time()
    # For GRPO, each update processes num_processes * num_generations episodes
    # So num_updates = total_env_steps / (num_processes * num_generations)
    num_updates = int(args.num_env_steps) // args.num_processes // num_samples
    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run + "-" + args.env_name
            wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)
        except ImportError:
            print("Warning: wandb is not installed. Install it with: pip install wandb")
            print("Continuing without wandb logging...")
            args.use_wandb = False

    # Print prompt only once at the start
    print("=" * 80)
    print("PROMPT:")
    print(qs)
    print("=" * 80)
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    num_explore = int(args.explore_portion*num_updates)
    prev_infos = []
    infos = []
    
    # Print interval for detailed output (only print every N updates)
    print_interval = 10  # Print detailed info every 10 updates
    
    # Create directory for saving maze images
    maze_images_dir = os.path.join(args.log_dir if hasattr(args, 'log_dir') and args.log_dir else './maze_images', 'episode_images')
    os.makedirs(maze_images_dir, exist_ok=True)
    print(f"Maze images will be saved to: {maze_images_dir}")
    
    def save_maze_image(obs, update_idx, proc_idx, gen_idx=None):
        """Save maze observation as an image file."""
        try:
            # Convert torch tensor to numpy if needed
            if isinstance(obs, torch.Tensor):
                # Move to CPU if on GPU
                obs_cpu = obs.cpu()
                # Handle batched observations
                if obs_cpu.dim() == 4:  # [num_processes, H, W, C]
                    obs_np = obs_cpu[proc_idx].numpy()
                elif obs_cpu.dim() == 3:  # [H, W, C]
                    obs_np = obs_cpu.numpy()
                else:
                    return  # Unexpected shape
            elif isinstance(obs, np.ndarray):
                if obs.ndim == 4:  # [num_processes, H, W, C]
                    obs_np = obs[proc_idx].copy()
                elif obs.ndim == 3:  # [H, W, C]
                    obs_np = obs.copy()
                else:
                    return  # Unexpected shape
            else:
                return  # Unknown type
            
            # Ensure values are in [0, 255] range and uint8
            # VecPyTorch converts to float, but values should still be in [0, 255] range
            if obs_np.dtype != np.uint8:
                obs_np = np.clip(obs_np, 0, 255).astype(np.uint8)
            
            # Ensure shape is (H, W, 3) - handle both (H, W, C) and (C, H, W) formats
            if len(obs_np.shape) == 3:
                if obs_np.shape[0] == 3 and obs_np.shape[-1] != 3:
                    # Likely (C, H, W) format, transpose to (H, W, C)
                    obs_np = np.transpose(obs_np, (1, 2, 0))
                elif obs_np.shape[-1] != 3:
                    return  # Not RGB and not transposable
            else:
                return  # Unexpected shape
            
            # Create PIL Image and save
            img = Image.fromarray(obs_np)
            if gen_idx is not None:
                filename = f"update_{update_idx:05d}_proc_{proc_idx:02d}_gen_{gen_idx:02d}.png"
            else:
                filename = f"update_{update_idx:05d}_proc_{proc_idx:02d}.png"
            filepath = os.path.join(maze_images_dir, filename)
            img.save(filepath)
        except Exception as e:
            print(f"Warning: Failed to save maze image for update {update_idx}, proc {proc_idx}: {e}")
    
    for j in tqdm(range(num_updates)):

        # GRPO: Generate full action sequences for each process
        # Reset environments to get initial observations
        obs = envs.reset()
        
        # Prepare storage: [num_processes, num_generations, ...]
        num_processes = args.num_processes
        
        # Save maze images for each process at the start of the episode
        for proc_idx in range(num_processes):
            save_maze_image(obs, j, proc_idx)
        
        with torch.no_grad():
            INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            INPUT_IDS[INPUT_IDS == 0] = 259
            # Generate num_samples completions per process
            output_ids_list, action_list, action_tokens_log_prob_list = actor_critic.act(obs, INPUT_IDS=INPUT_IDS)
        num_generations = num_samples
        
        # Stack output_ids: [num_processes, num_generations, 2*max_new_tokens]
        # output_ids_list is a list of [num_processes, 2*max_new_tokens] tensors
        output_ids_batch = torch.stack(output_ids_list, dim=1)  # [num_processes, num_generations, 2*max_new_tokens]
        
        # Stack action_tokens_log_prob: [num_processes, num_generations, max_tokens]
        # action_tokens_log_prob_list is a list of [num_processes, max_tokens] tensors
        action_log_probs_batch = torch.stack(action_tokens_log_prob_list, dim=1)  # [num_processes, num_generations, max_tokens]
        
        # Ensure max_tokens dimension matches storage
        max_tokens_storage = action_log_probs_batch.size(2)
        
        # Create token masks (non-zero tokens)
        # output_ids_batch: [num_processes, num_generations, 2*max_new_tokens]
        # We need masks for tokens after the first one: [num_processes, num_generations, max_tokens]
        # max_tokens = 2*max_new_tokens - 2 (as per storage definition)
        max_tokens = 2 * args.max_new_tokens - 2
        token_masks_batch = (output_ids_batch[:, :, 1:max_tokens+1] != 0).long()  # [num_processes, num_generations, max_tokens]
        
        # Ensure action_log_probs_batch matches max_tokens
        if action_log_probs_batch.size(2) > max_tokens:
            action_log_probs_batch = action_log_probs_batch[:, :, :max_tokens]
        elif action_log_probs_batch.size(2) < max_tokens:
            padding = torch.zeros(
                action_log_probs_batch.size(0), action_log_probs_batch.size(1),
                max_tokens - action_log_probs_batch.size(2),
                dtype=action_log_probs_batch.dtype, device=action_log_probs_batch.device
            )
            action_log_probs_batch = torch.cat([action_log_probs_batch, padding], dim=2)
        
        # Execute action sequences and collect rewards
        rewards_batch = torch.zeros(num_processes, num_generations, 1)
        actions_batch = torch.zeros(num_processes, num_generations, 1, dtype=torch.long)
        
        initial_obs_per_proc = []
        if isinstance(obs, np.ndarray):
            if obs.ndim == 4:  # [num_processes, H, W, C]
                for i in range(num_processes):
                    initial_obs_per_proc.append(obs[i])
            else:
                # Single observation, replicate for all processes
                for i in range(num_processes):
                    initial_obs_per_proc.append(obs)
        else:
            # Handle other formats
            for i in range(num_processes):
                initial_obs_per_proc.append(obs)
        
        # For each process and generation, execute the action sequence
        for proc_idx in range(num_processes):
            for gen_idx in range(num_generations):
                # Get action sequence for this generation
                action_sequence = action_list[gen_idx][proc_idx]  # List of action indices
                
                # Reset environment for this process (each generation starts fresh)
                # VecPyTorch wraps the base vectorized env, need to unwrap
                base_vec_env = envs.venv if hasattr(envs, 'venv') else envs
                if hasattr(base_vec_env, 'envs'):
                    # Vectorized environment - get individual env
                    env = base_vec_env.envs[proc_idx]
                    # Unwrap any additional wrappers if needed
                    while hasattr(env, 'env'):
                        env = env.env
                    proc_obs, _ = env.reset()
                    if isinstance(proc_obs, tuple):
                        proc_obs = proc_obs[0]
                else:
                    # Single environment - reset the whole vectorized env and take first
                    proc_obs, _ = envs.reset()
                    if isinstance(proc_obs, tuple):
                        proc_obs = proc_obs[0]
                    if proc_obs.ndim == 4:  # [num_processes, H, W, C]
                        proc_obs = proc_obs[proc_idx]
                
                # Execute action sequence
                try:
                    final_obs, total_reward, terminated, truncated, final_info, step_count = grpo_act(
                        env, action_sequence
                    )
                except Exception as e:
                    print(f"Error executing action sequence for proc {proc_idx}, gen {gen_idx}: {e}")
                    total_reward = 0.0
                    terminated = False
                    truncated = False
                    final_info = {}
                    step_count = 0
                
                rewards_batch[proc_idx, gen_idx, 0] = total_reward
                # Store first action as placeholder (not used in GRPO training)
                if len(action_sequence) > 0:
                    actions_batch[proc_idx, gen_idx, 0] = action_sequence[0]
                else:
                    actions_batch[proc_idx, gen_idx, 0] = 0
                
                # Track episode completion (only last generation)
                if gen_idx == num_generations - 1:
                    is_success = total_reward > 0
                    episode_rewards.append(float(total_reward))
                    if is_success:
                        episode_success_rate.append(1)
                    else:
                        episode_success_rate.append(0)
        
        # Store in GRPO rollout storage
        masks = torch.ones(num_processes, 1)  # All episodes continue (we reset between generations)
        bad_masks = torch.ones(num_processes, 1)
        
        rollouts.insert(
            obs=obs,
            output_ids=output_ids_batch,
            actions=actions_batch,
            action_log_probs=action_log_probs_batch,
            rewards=rewards_batch,
            masks=masks,
            bad_masks=bad_masks,
            token_masks=token_masks_batch
        )
        
        # Only print detailed info every N updates
        should_print = (j % print_interval == 0) or (j == 0)
        
        if should_print:
            print("****** iteration number:{} ******".format(j))
            print("Rewards shape: {}".format(rewards_batch.shape))
            print("Mean reward per generation: {:.4f}".format(rewards_batch.mean().item()))
        
        # GRPO Update: Use the GRPO agent's update method
        # This handles group-relative advantages, PPO clipping, and all training logic
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        
        lr_scheduler.step()
        
        if len(episode_rewards) > 1:
            # For GRPO, each update processes num_processes * num_generations episodes
            total_num_steps = (j + 1) * args.num_processes * num_samples
            end = time.time()

            # Only print detailed stats every N updates
            if should_print:
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}, dist_entropy {:.4f}, action_loss {:.4f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), np.mean(episode_success_rate),
                            dist_entropy, action_loss))
            if args.use_wandb:
                wandb.log({"iteration": j,
                        "num_timesteps": total_num_steps,
                        "FPS": int(total_num_steps / (end - start)),
                        "episode_reward.mean": np.mean(episode_rewards),
                        "episode_reward.median": np.median(episode_rewards),
                        "episode_reward.min": np.min(episode_rewards),
                        "episode_reward.min": np.max(episode_rewards),
                        "episode_success_rate.mean": np.mean(episode_success_rate),
                        "episode_action_tokens_log_prob.mean": np.mean(episode_action_tokens_log_prob),
                        "distribution_entropy": dist_entropy,
                        "action.loss": action_loss,
                        "reward.max": rollouts.rewards[0].max().item(),
                        "reward.min": rollouts.rewards[0].min().item(),
                        "reward.mean": rollouts.rewards[0].mean().item(),
                        "reward.std": rollouts.rewards[0].std().item(),
                        "reward.median": rollouts.rewards[0].median().item(),})

if __name__ == "__main__":
    main()


