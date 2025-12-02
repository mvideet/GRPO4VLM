# -*- coding: utf-8 -*-
"""
Inference Utilities for Trained Maze-Solving Model
"""

from PIL import Image
from typing import List, Tuple
from config import MazeGRPOConfig
from maze_dataset import MazeDatasetGenerator
from maze_env import grpo_act
from prompts import SYSTEM_PROMPT, create_maze_prompt
from rewards import parse_actions_from_completion


def run_inference(model, tokenizer, image: Image.Image, maze_size: Tuple[int, int],
                  agent_pos: List[int], goal_pos: List[int]) -> str:
    """Run inference on a single maze image."""
    from unsloth import FastVisionModel
    
    FastVisionModel.for_inference(model)
    
    prompt_text = create_maze_prompt(list(maze_size), agent_pos, goal_pos)
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        use_cache=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def test_trained_model(model_path: str = "maze_solver_lora"):
    """Test the trained model on a sample maze."""
    from unsloth import FastVisionModel
    
    print("Loading trained model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    
    # Generate a test maze
    config = MazeGRPOConfig()
    generator = MazeDatasetGenerator(config)
    maze_data = generator.generate_single_maze((5, 5), seed=42)
    
    print("\nRunning inference...")
    response = run_inference(
        model, tokenizer,
        maze_data['image'],
        maze_data['maze_size'],
        maze_data['agent_pos'],
        maze_data['goal_pos']
    )
    
    print("\n" + "=" * 60)
    print("MODEL RESPONSE:")
    print("=" * 60)
    print(response)
    
    # Parse and execute actions
    actions = parse_actions_from_completion(response)
    if actions:
        print(f"\nParsed actions: {actions}")
        print(f"Action names: {[['N','S','E','W'][a] for a in actions]}")
        
        # Execute in environment
        env = maze_data['env']
        env.reset()
        obs, reward, terminated, truncated, info, steps = grpo_act(env, actions)
        
        print(f"\nExecution result:")
        print(f"  Steps taken: {steps}")
        print(f"  Total reward: {reward:.2f}")
        print(f"  Solved: {terminated}")
    else:
        print("\nCould not parse actions from response.")


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "maze_solver_lora"
    test_trained_model(model_path)

