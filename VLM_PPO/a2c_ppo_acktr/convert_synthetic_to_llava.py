"""
Convert synthetic maze-solving data to LLaVA training format.
"""
import json
import os
import uuid
from typing import List, Dict, Any


def convert_synthetic_to_llava(
    input_path: str = "synthetic_data.json",
    output_path: str = "synthetic_data_llava.json",
    include_image: bool = False,
    image_path: str = None
) -> None:
    """
    Convert synthetic maze-solving data to LLaVA format.
    
    Args:
        input_path: Path to synthetic data JSON file
        output_path: Path to save LLaVA-formatted data
        include_image: Whether to include image field (for multimodal training)
        image_path: Path to image file if include_image is True
    """
    # Load synthetic data
    with open(input_path, 'r') as f:
        synthetic_data = json.load(f)
    
    llava_data = []
    
    for i, sample in enumerate(synthetic_data):
        # Create the prompt for the human
        human_prompt = """Solve this maze by thinking step by step and providing your actions.
        
Think through the maze navigation strategy, then output your actions in the following JSON format:
{
  "thoughts": "...",
  "actions": ["up","down","left","right", ...]
}"""
        
        # Create the assistant response (thoughts + actions in JSON format)
        assistant_response = json.dumps({
            "thoughts": sample.get("thoughts", ""),
            "actions": sample.get("actions", [])
        }, indent=2)
        
        # Create LLaVA format entry
        entry = {
            "id": str(uuid.uuid4()),
            "conversations": [
                {
                    "from": "human",
                    "value": human_prompt
                },
                {
                    "from": "gpt",
                    "value": assistant_response
                }
            ]
        }
        
        # Optionally add image field
        if include_image:
            if image_path:
                entry["image"] = image_path
            else:
                # Use a placeholder or default image
                entry["image"] = "placeholder_maze.png"
        
        llava_data.append(entry)
    
    # Save converted data
    with open(output_path, 'w') as f:
        json.dump(llava_data, f, indent=2)
    
    print(f"Converted {len(llava_data)} samples from {input_path} to {output_path}")
    print(f"Sample entry:")
    print(json.dumps(llava_data[0], indent=2))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert synthetic data to LLaVA format")
    parser.add_argument("--input", type=str, default="synthetic_data.json",
                        help="Input synthetic data JSON file")
    parser.add_argument("--output", type=str, default="synthetic_data_llava.json",
                        help="Output LLaVA-formatted JSON file")
    parser.add_argument("--include-image", action="store_true",
                        help="Include image field in output (for multimodal training)")
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path to image file if including images")
    
    args = parser.parse_args()
    
    convert_synthetic_to_llava(
        input_path=args.input,
        output_path=args.output,
        include_image=args.include_image,
        image_path=args.image_path
    )

