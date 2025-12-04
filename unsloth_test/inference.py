# -*- coding: utf-8 -*-
"""
Inference Utilities for Trained Maze-Solving Model
"""

from PIL import Image
from typing import List, Tuple
import numpy as np
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
    
    # Display maze information
    print("\n" + "=" * 60)
    print("MAZE INFORMATION:")
    print("=" * 60)
    print(f"  Size: {maze_data['maze_size']}")
    print(f"  Agent position: {maze_data['agent_pos']}")
    print(f"  Goal position: {maze_data['goal_pos']}")
    print(f"  Seed: {maze_data['seed']}")
    
    # Save and display maze image
    import os
    maze_image_path = "/tmp/test_maze.png"
    maze_data['image'].save(maze_image_path)
    print(f"\n  Maze image saved to: {maze_image_path}")
    
    # Try to display the image
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        img = mpimg.imread(maze_image_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Maze {maze_data['maze_size'][0]}x{maze_data['maze_size'][1]} - Start: {maze_data['agent_pos']}, Goal: {maze_data['goal_pos']}")
        plt.tight_layout()
        plt.savefig(maze_image_path, bbox_inches='tight', dpi=150)
        print(f"  Displaying maze image...")
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure display
    except Exception as e:
        print(f"  Could not display image (matplotlib not available or no display): {e}")
        print(f"  Image saved at: {maze_image_path}")
    
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
        action_names = [['N','S','E','W'][a] for a in actions]
        print(f"Action names: {action_names}")
        print(f"Action sequence: {' -> '.join(action_names)}")
        
        # Execute in environment
        env = maze_data['env']
        env.reset()
        obs, reward, terminated, truncated, info, steps = grpo_act(env, actions)
        
        print(f"\n" + "=" * 60)
        print("EXECUTION RESULT:")
        print("=" * 60)
        print(f"  Steps taken: {steps}")
        print(f"  Total reward: {reward:.2f}")
        print(f"  Solved: {terminated}")
        print(f"  Final position: {info.get('agent_pos', 'N/A')}")
        print(f"  Goal position: {maze_data['goal_pos']}")
        
        # Save final state image showing the path taken
        try:
            import matplotlib.pyplot as plt
            
            # Get the final rendered state
            final_img = np.array(obs) if isinstance(obs, np.ndarray) else np.array(env.render())
            
            result_image_path = "/tmp/maze_result.png"
            plt.figure(figsize=(10, 5))
            
            # Show original maze
            plt.subplot(1, 2, 1)
            original_img = np.array(maze_data['image'])
            plt.imshow(original_img)
            plt.axis('off')
            plt.title("Original Maze")
            
            # Show final state with path
            plt.subplot(1, 2, 2)
            plt.imshow(final_img)
            plt.axis('off')
            status = "✓ SOLVED!" if terminated else "✗ Not solved"
            plt.title(f"After Execution - {status}")
            
            plt.tight_layout()
            plt.savefig(result_image_path, bbox_inches='tight', dpi=150)
            print(f"\n  Result visualization saved to: {result_image_path}")
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as e:
            print(f"  Could not create result visualization: {e}")
    else:
        print("\nCould not parse actions from response.")


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "maze_solver_lora"
    test_trained_model(model_path)

