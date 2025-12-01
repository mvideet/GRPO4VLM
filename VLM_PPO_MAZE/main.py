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

import gymnasium as gym
import gym_maze
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model
from a2c_ppo_acktr.curriculum import MazeCurriculum

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

from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

import warnings
warnings.filterwarnings("ignore")

def main():
    args = get_args()

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
    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    # Initialize curriculum learning if enabled
    curriculum = None
    if args.use_curriculum and ("maze" in args.env_name.lower() or "gym_maze" in args.env_name.lower()):
        curriculum = MazeCurriculum(
            start_size=args.curriculum_start_size,
            end_size=args.curriculum_end_size,
            progression_criterion=args.curriculum_progression,
            success_rate_threshold=args.curriculum_success_threshold,
            min_episodes_per_size=args.curriculum_min_episodes,
            updates_per_size=args.curriculum_updates_per_size,
        )
        print(f"Curriculum learning enabled: {args.curriculum_start_size}x{args.curriculum_start_size} -> {args.curriculum_end_size}x{args.curriculum_end_size}")
        print(f"Progression criterion: {args.curriculum_progression}")
        current_env_name = curriculum.get_current_env_name()
        current_maze_size = curriculum.get_current_size()
    else:
        current_env_name = args.env_name
        current_maze_size = None

    # Create maze environment
    if "maze" in args.env_name.lower() or "gym_maze" in args.env_name.lower() or (curriculum is not None):
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

    projection_f = partial(text_projection, env_name=prompt_env_name)

    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS,
                             args=args)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    agent = algo.PPO(
            actor_critic,
            optimizer,
            accelerator,
            args.clip_param,
            args.ppo_epoch,
            args.mini_batch_size,
            args.value_loss_coef,
            args.entropy_coef,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, args.max_new_tokens)

    _, output_ids, action, action_log_prob, action_tokens_log_prob = actor_critic.act(obs, INPUT_IDS = INPUT_IDS)
    print("action:{}".format(action))
    print("action_log_prob:{}".format(action_log_prob))
    print("action_tokens_log_prob:{}".format(action_tokens_log_prob))

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)
    
    # Step-by-step metrics tracking (one action per step)
    step_rewards = deque(maxlen=args.eval_num_per_episode * 100)  # Track individual step rewards
    step_action_log_probs = deque(maxlen=args.eval_num_per_episode * 100)  # Track per-step action log probs
    step_values = deque(maxlen=args.eval_num_per_episode * 100)  # Track per-step values
    step_counts_per_episode = deque(maxlen=args.eval_num_per_episode)  # Track steps per episode
    running_step_count = torch.zeros(args.num_processes).flatten()  # Track current step count per process

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run + "-" + args.env_name
        wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)

    print(qs)
    print("=== Step-by-Step PPO Training: One action per environment step ===")
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    num_explore = int(args.explore_portion*num_updates)
    prev_infos = []
    infos = []
    
    # Track curriculum progression
    last_curriculum_check = 0
    curriculum_check_interval = 10  # Check every 10 updates
    
    for j in tqdm(range(num_updates)):

        # Check for curriculum progression
        if curriculum is not None and (j - last_curriculum_check) >= curriculum_check_interval:
            curriculum_info = curriculum.get_progress_info()
            print(f"\n=== Curriculum Status ===")
            print(f"Current maze size: {curriculum_info['current_size']}x{curriculum_info['current_size']}")
            print(f"Progress: {curriculum_info['progress_percentage']:.1f}% ({curriculum_info['current_size_idx']+1}/{curriculum_info['total_sizes']})")
            print(f"Episodes: {curriculum_info['episode_count']}, Successes: {curriculum_info['success_count']}")
            if curriculum_info['episode_count'] > 0:
                print(f"Success rate: {curriculum_info['current_success_rate']:.3f} (threshold: {curriculum_info['threshold']})")
            
            if curriculum.should_progress():
                print(f"\n*** PROGRESSING TO NEXT MAZE SIZE ***")
                if curriculum.progress():
                    new_size = curriculum.get_current_size()
                    new_env_name = curriculum.get_current_env_name()
                    print(f"New maze size: {new_size}x{new_size}")
                    print(f"New environment: {new_env_name}")
                    
                    # Close old environments
                    envs.close()
                    
                    # Create new environments with new size
                    envs = make_vec_envs(new_env_name, args.seed, args.num_processes,
                                         args.gamma, None, device, False, 1, maze_size=new_size)
                    
                    # Reset observation space in rollouts
                    obs = envs.reset()
                    rollouts.obs[0].copy_(obs)
                    
                    # Update prompt for new environment
                    qs = get_prompt(new_env_name, args.action_only_prompt, None)
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    
                    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                    INPUT_IDS[INPUT_IDS == 0] = 259
                    
                    # Update projection function for new environment
                    projection_f = partial(text_projection, env_name=new_env_name)
                    actor_critic.projection_f = projection_f
                    
                    print(f"Environment updated successfully!")
            
            last_curriculum_check = j

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
                value, output_id, action, action_log_prob, action_tokens_log_prob = actor_critic.act(
                        rollouts.obs[step], INPUT_IDS = INPUT_IDS)
            text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())))
            prev_infos = copy.deepcopy(infos)
            obs, reward, done, infos = envs.step(action)

            # Use current environment name (may change with curriculum)
            prompt_env_name = curriculum.get_current_env_name() if curriculum is not None else args.env_name
            qs = get_prompt(prompt_env_name, args.action_only_prompt, infos)
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            # Track step-by-step metrics (one action per step)
            for i in range(args.num_processes):
                step_rewards.append(reward[i].item())
                step_action_log_probs.append(action_log_prob[i].item())
                step_values.append(value[i].item())
                running_step_count[i] += 1

            running_episode_rewards += reward.flatten()
            for i, d, r in zip(range(args.num_processes), done, reward):
                if d:
                    episode_reward = running_episode_rewards[i].item()
                    episode_rewards.append(episode_reward)
                    
                    # Check if agent reached goal by comparing positions
                    info = infos[i] if isinstance(infos, list) else infos
                    agent_pos = info.get('agent_pos') if isinstance(info, dict) else None
                    goal_pos = info.get('goal_pos') if isinstance(info, dict) else None
                    
                    if agent_pos is not None and goal_pos is not None:
                        # Success if agent is at goal (within 0.5 units tolerance)
                        distance = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))
                        is_success = distance < 0.5
                    else:
                        # Fallback: check if episode ended with positive reward (goal reached)
                        is_success = episode_reward > 0
                    
                    if is_success:
                        episode_success_rate.append(1)
                    else:
                        episode_success_rate.append(0)
                    episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                    # Record step count for this episode
                    step_counts_per_episode.append(running_step_count[i].item())
                    running_episode_rewards[i] = 0
                    running_step_count[i] = 0
                    
                    # Record episode for curriculum learning
                    if curriculum is not None:
                        curriculum.record_episode(is_success)
            # bad_mask is a legacy implementation of the storage.py file
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, output_id, action,
                            action_log_prob, value, reward, masks, bad_masks)
        print("****** iteration number:{} (Step-by-Step PPO: {} steps collected, one action per step) ******".format(j, args.num_steps))
        print("prompt:{}".format(prompt))
        print("text_action:{}".format(text_action))
        print("current observation:{}".format(prev_infos))
        print("ground truth:{}".format(infos))
        print("action log prob:{}".format(action_log_prob))
        print("action tokens log prob:{}".format(action_tokens_log_prob))
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        lr_scheduler.step()

        rollouts.after_update()
        
        # Record update for curriculum learning (if using updates criterion)
        if curriculum is not None and args.curriculum_progression == "updates":
            curriculum.record_update()
        if len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            avg_steps_per_episode = np.mean(step_counts_per_episode) if len(step_counts_per_episode) > 0 else 0.0
            print(
                "Updates {}, num timesteps {}, FPS {} \n"
                "Step-by-Step PPO: Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}, avg steps/episode {:.1f}\n"
                "Step-level metrics: mean step reward {:.4f}, mean step value {:.4f}, mean step action_log_prob {:.4f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_success_rate),
                        avg_steps_per_episode,
                        np.mean(step_rewards) if len(step_rewards) > 0 else 0.0,
                        np.mean(step_values) if len(step_values) > 0 else 0.0,
                        np.mean(step_action_log_probs) if len(step_action_log_probs) > 0 else 0.0))
            if args.use_wandb:
                # Step-by-step metrics (one action per step)
                step_metrics = {
                    "step.reward.mean": np.mean(step_rewards) if len(step_rewards) > 0 else 0.0,
                    "step.reward.std": np.std(step_rewards) if len(step_rewards) > 0 else 0.0,
                    "step.action_log_prob.mean": np.mean(step_action_log_probs) if len(step_action_log_probs) > 0 else 0.0,
                    "step.value.mean": np.mean(step_values) if len(step_values) > 0 else 0.0,
                    "step.value.std": np.std(step_values) if len(step_values) > 0 else 0.0,
                    "step.count_per_episode.mean": np.mean(step_counts_per_episode) if len(step_counts_per_episode) > 0 else 0.0,
                }
                
                # Rollout-level step metrics (from current rollout)
                rollout_step_metrics = {
                    "rollout.step.reward.max": rollouts.rewards.max().item(),
                    "rollout.step.reward.min": rollouts.rewards.min().item(),
                    "rollout.step.reward.mean": rollouts.rewards.mean().item(),
                    "rollout.step.reward.std": rollouts.rewards.std().item(),
                    "rollout.step.reward.median": rollouts.rewards.median().item(),
                    "rollout.step.return.max": rollouts.returns[:-1].max().item(),
                    "rollout.step.return.min": rollouts.returns[:-1].min().item(),
                    "rollout.step.return.mean": rollouts.returns[:-1].mean().item(),
                    "rollout.step.return.std": rollouts.returns[:-1].std().item(),
                    "rollout.step.value.max": rollouts.value_preds[:-1].max().item(),
                    "rollout.step.value.min": rollouts.value_preds[:-1].min().item(),
                    "rollout.step.value.mean": rollouts.value_preds[:-1].mean().item(),
                    "rollout.step.value.std": rollouts.value_preds[:-1].std().item(),
                    "rollout.step.action_log_prob.mean": rollouts.action_log_probs.mean().item(),
                    "rollout.step.action_log_prob.std": rollouts.action_log_probs.std().item(),
                }
                
                # Episode-level metrics (aggregated from completed episodes)
                episode_metrics = {
                    "episode.reward.mean": np.mean(episode_rewards),
                    "episode.reward.median": np.median(episode_rewards),
                    "episode.reward.min": np.min(episode_rewards),
                    "episode.reward.max": np.max(episode_rewards),
                    "episode.success_rate.mean": np.mean(episode_success_rate),
                    "episode.action_tokens_log_prob.mean": np.mean(episode_action_tokens_log_prob),
                }
                
                # Training metrics
                training_metrics = {
                    "iteration": j,
                    "num_timesteps": total_num_steps,
                    "FPS": int(total_num_steps / (end - start)),
                    "distribution_entropy": dist_entropy,
                    "value.loss": value_loss,
                    "action.loss": action_loss,
                }
                
                wandb.log({**step_metrics, **rollout_step_metrics, **episode_metrics, **training_metrics})
            
            # Log curriculum information
            if curriculum is not None:
                curriculum_info = curriculum.get_progress_info()
                wandb.log({
                    "curriculum.current_size": curriculum_info['current_size'],
                    "curriculum.progress_percentage": curriculum_info['progress_percentage'],
                    "curriculum.current_success_rate": curriculum_info['current_success_rate'],
                    "curriculum.episode_count": curriculum_info['episode_count'],
                    "curriculum.update_count": curriculum_info['update_count'],
                })

if __name__ == "__main__":
    main()


