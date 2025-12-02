# -*- coding: utf-8 -*-
"""
Main Training Script for Maze-Solving VLM with GRPO
"""

import os

# Set up vLLM standby for better memory efficiency
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from config import MazeGRPOConfig
from maze_dataset import MazeDatasetGenerator
from prompts import make_conversation
from rewards import formatting_reward_func, maze_execution_reward_func
from curriculum import CurriculumLearningCallback, CurriculumTrainerCallback


def main():
    """Main training function."""
    from unsloth import FastVisionModel
    from trl import GRPOConfig, GRPOTrainer
    
    config = MazeGRPOConfig()
    
    print("=" * 60)
    print("MAZE VLM GRPO TRAINING")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading model...")
    import wandb
    wandb.init(
        project="maze-vlm-grpo",
        name="maze-solver-run-1",
        config={
            "model": "gemma-3-4b-it",
            "lora_r": 16,
            "learning_rate": 5e-6,
            "maze_sizes": [(5,5), (7,7), (10,10)],
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "num_generations": 4,
            "max_prompt_length": 1024,
            "max_completion_length": 512,
            "max_steps": 200,
            "save_steps": 50,
            "format_reward": 1.0,
            "solve_reward": 10.0,
            "efficiency_bonus": 0.1,
            "partial_credit_weight": 1.0,
            "use_curriculum": config.use_curriculum,
            "curriculum_steps_per_level": config.curriculum_steps_per_level,
            "curriculum_start_level": config.curriculum_start_level,
        }
    )
    
    model, tokenizer = FastVisionModel.from_pretrained(
        config.model_name,
        load_in_4bit=config.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Don't finetune vision (maze images are simple)
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
    )
    
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_r}")
    
    # -------------------------------------------------------------------------
    # 2. Generate Dataset
    # -------------------------------------------------------------------------
    print("\n[2/5] Generating maze dataset...")
    
    generator = MazeDatasetGenerator(config)
    dataset = generator.generate_dataset()
    
    print(f"  Generated {len(dataset)} mazes")
    print(f"  Maze sizes: {config.maze_sizes}")
    
    # Save all maze images
    generator.save_maze_images(dataset, output_dir="maze_images")
    
    # -------------------------------------------------------------------------
    # 3. Prepare Dataset for Training
    # -------------------------------------------------------------------------
    print("\n[3/5] Preparing dataset...")
    def prepare_example(example):
        conv = make_conversation(example)
        prompt_text = tokenizer.apply_chat_template(
            conv["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )
        result = {
            "prompt": prompt_text,
            "image": conv["image"],
            "maze_size": conv["maze_size"],
            "agent_pos": conv["agent_pos"],
            "goal_pos": conv["goal_pos"],
            "seed": conv["seed"]
        }
        
        # Preserve difficulty_level if present (for curriculum learning)
        if "difficulty_level" in conv:
            result["difficulty_level"] = conv["difficulty_level"]
        
        return result
    
    train_dataset = dataset.map(prepare_example)
    
    # Initialize curriculum learning
    curriculum_callback = CurriculumLearningCallback(config, train_dataset)
    
    # Start with initial curriculum level (easiest mazes only)
    if config.use_curriculum:
        initial_dataset = curriculum_callback.get_current_dataset(0)
        train_dataset = initial_dataset
        level_info = curriculum_callback.get_current_level_info()
        print(f"\n🎓 Curriculum Learning: Starting at Level {level_info['current_level']} "
              f"(Maze size: {level_info['current_maze_size']})")
        print(f"  Will advance every {config.curriculum_steps_per_level} steps")
    else:
        print("\n  Curriculum learning: DISABLED (using all mazes from start)")
    
    print(f"  Dataset ready with {len(train_dataset)} examples")
    
    # Show sample prompt
    print("\n  Sample prompt (truncated):")
    print("  " + "-" * 50)
    sample_prompt = train_dataset[0]["prompt"][:500]
    print(f"  {sample_prompt}...")
    print("\n[4/5] Configuring GRPO trainer...")
    
    training_args = GRPOConfig(
        learning_rate=config.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_steps=config.max_steps,
        save_steps=config.save_steps,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="maze_grpo_outputs",
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type='dr_grpo',
    )
    
    # Create trainer with both reward functions
    # Create callbacks list properly
    trainer_callbacks = []
    curriculum_trainer_callback = None
    if config.use_curriculum:
        curriculum_trainer_callback = CurriculumTrainerCallback(curriculum_callback)
        trainer_callbacks.append(curriculum_trainer_callback)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[
            formatting_reward_func,
            maze_execution_reward_func
        ],
        train_dataset=train_dataset,
        callbacks=trainer_callbacks if trainer_callbacks else None,
    )
        
    # Set trainer reference in callback after creation
    if curriculum_trainer_callback is not None:
        curriculum_trainer_callback.set_trainer(trainer)
    
    print("  Training configuration:")
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Batch size: {config.batch_size}")
    print(f"    Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"    Max steps: {config.max_steps}")
    
    # -------------------------------------------------------------------------
    # 5. Train!
    # -------------------------------------------------------------------------
    print("\n[5/5] Starting training...")
    print("=" * 60)
    print("Watch the 'reward' column - it should increase over time!")
    print("Be patient - first 50-100 steps may show low rewards.")
    print("=" * 60)
    
    trainer.train()
    
    # -------------------------------------------------------------------------
    # Save Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training complete! Saving model...")
    
    model.save_pretrained("maze_solver_lora")
    tokenizer.save_pretrained("maze_solver_lora")
    
    print("Model saved to: maze_solver_lora/")
    print("=" * 60)


if __name__ == "__main__":
    main()

