import openai
import numpy as np
import json
import os
import time
from typing import Optional, Dict, Any

# Load API key from environment variable (more secure)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")

client = openai.OpenAI(api_key=api_key)

prompt = """Generate a synthetic example of a maze-solving trajectory in the following JSON format:

{
  "thoughts": "...",
  "actions": ["up","down","left","right", ...]
}

Instructions:
- Do NOT output a maze.
- "thoughts" should contain a coherent chain-of-thought describing how an agent *would* reason through a top-down maze, without referencing any map.
- "actions" should be a realistic sequence of moves that would plausibly navigate a maze.
- Action lists must be 3–40 steps, randomly.
- Every sample must be unique.
- Only output the JSON object, no extra text."""


def validate_data(data: Dict[str, Any]) -> bool:
    """Validate that generated data matches expected format."""
    if not isinstance(data, dict):
        return False
    if "thoughts" not in data or "actions" not in data:
        return False
    if not isinstance(data["actions"], list):
        return False
    if not (3 <= len(data["actions"]) <= 40):
        return False
    # Check that all actions are valid
    valid_actions = {"up", "down", "left", "right"}
    if not all(action.lower() in valid_actions for action in data["actions"]):
        return False
    return True


def generate_synthetic_data(max_retries: int = 3, retry_delay: float = 1.0) -> Optional[Dict[str, Any]]:
    """Generate a single synthetic data sample with retry logic."""
    temperature = np.random.uniform(0.8, 1.2)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = response.choices[0].message.content
            
            # Try to parse JSON
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Sometimes the response has markdown code blocks
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
            
            # Validate data format
            if validate_data(data):
                return data
            else:
                print(f"Invalid data format (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
                
        except openai.RateLimitError:
            print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting...")
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            continue
        except Exception as e:
            print(f"Error generating data (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None
    
    return None


def generate_dataset(num_samples: int = 1000, output_path: str = "synthetic_data.json"):
    """Generate a dataset of synthetic maze-solving trajectories."""
    data_list = []
    successful = 0
    failed = 0
    print(f"Generating {num_samples} synthetic samples...")
    
    for i in range(num_samples):
        sample = generate_synthetic_data()
        if sample is not None:
            data_list.append(sample)
            successful += 1
        else:
            failed += 1
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{num_samples} | Successful: {successful} | Failed: {failed}")
        
        time.sleep(0.1)
    
    print(f"\nGeneration complete!")
    print(f"Total: {num_samples} | Successful: {successful} | Failed: {failed}")
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(data_list, f, indent=2)
    
    print(f"Saved {len(data_list)} samples to {output_path}")
    return data_list


if __name__ == "__main__":
    generate_dataset(num_samples=1000)

