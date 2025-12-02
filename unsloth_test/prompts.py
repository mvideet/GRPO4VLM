# -*- coding: utf-8 -*-
"""
Prompt Templates for Maze Solving
"""

from typing import List, Dict

# Delimiter tags for structured output
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
ACTIONS_START = "<ACTIONS>"
ACTIONS_END = "</ACTIONS>"

SYSTEM_PROMPT = """You are a maze-solving AI. You will be shown an image of a maze and must navigate from the green circle (start) to the red circle (goal).

The maze uses a grid system where:
- Black cells are walls (impassable)
- White cells are paths (walkable)
- Green circle = your current position (agent)
- Red circle = goal position

Available actions: UP, DOWN, LEFT, RIGHT

Output format:
1. First, provide your reasoning about the maze layout and optimal path inside {reasoning_start} and {reasoning_end} tags.
2. Then, provide your action sequence as a comma-separated list inside {actions_start} and {actions_end} tags.

Example output:
{reasoning_start}
I can see a 5x5 maze. I'm at the top-left corner and need to reach the bottom-right. 
There's a wall blocking the direct path, so I need to go around...
{reasoning_end}
{actions_start}
DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT
{actions_end}
""".format(
    reasoning_start=REASONING_START,
    reasoning_end=REASONING_END,
    actions_start=ACTIONS_START,
    actions_end=ACTIONS_END
)


def create_maze_prompt(maze_size: List[int], agent_pos: List[int], goal_pos: List[int]) -> str:
    """Create the user prompt for a specific maze."""
    return (
        f"Solve this {maze_size[0]}x{maze_size[1]} maze. "
        f"You are at position ({agent_pos[0]}, {agent_pos[1]}) and need to reach ({goal_pos[0]}, {goal_pos[1]}). "
        f"Provide your reasoning between {REASONING_START} and {REASONING_END}, "
        f"then your action sequence between {ACTIONS_START} and {ACTIONS_END}."
    )


def make_conversation(example: Dict) -> Dict:
    """Convert maze data to conversation format for VLM."""
    # Ensure all list fields are consistently lists (not tuples)
    maze_size = list(example['maze_size']) if not isinstance(example['maze_size'], list) else example['maze_size']
    agent_pos = list(example['agent_pos']) if not isinstance(example['agent_pos'], list) else example['agent_pos']
    goal_pos = list(example['goal_pos']) if not isinstance(example['goal_pos'], list) else example['goal_pos']
    
    text_content = create_maze_prompt(maze_size, agent_pos, goal_pos)
    
    prompt = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_content}
            ]
        }
    ]
    
    result = {
        "prompt": prompt,
        "image": example["image"],
        "maze_size": maze_size,
        "agent_pos": agent_pos,
        "goal_pos": goal_pos,
        "seed": int(example["seed"])
    }
    
    # Preserve difficulty_level if present (for curriculum learning)
    if "difficulty_level" in example:
        result["difficulty_level"] = example["difficulty_level"]
    
    return result

