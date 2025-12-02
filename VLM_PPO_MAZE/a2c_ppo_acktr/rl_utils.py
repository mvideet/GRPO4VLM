import torch
import random
import json
import re
from typing import List

def get_prompt(env_name, action_only=False, infos=None):
    """
        This function defines the prompt for the text-to-action task, depending on the environments
        env_name: determines the prompts for each environment
        action_only: if True, only include action in prompt (no thoughts)
        info: additional information that can be added to the prompt, if none, then use the default prompt
    """
    if 'maze' in env_name.lower() or 'gym_maze' in env_name.lower():
        # Maze environment prompt
        qs = (
            "You are an extremely smart maze solver. You are observing a top-down view of the maze. "
            "Your goal is to move from the current start position to the goal position, which is shown as a red square. "
            "You need to solve the entire maze in one shot without getting stuck. "
            "First, think step-by-step about the complete path you will take. "
            "After you have finished thinking, output the full trajectory as a sequence of actions. "
            "You can choose between the four directions: ['up', 'right', 'left', 'down']. "
            "Your response MUST be a valid JSON object in the following format. Once again ensure that you write a JSON object with thoughts and actions:\n"
            "{\n"
            '  "thoughts": "{first think carefully through the full path you will take}",\n'
            '  "actions": ["up", "right", "left", "..."]\n'
            "}"
        )
    elif env_name == 'gym_cards/NumberLine-v0':
        qs = "You are playing a game called number line. You will see a target number and a current number in the image. "
        qs = qs + "And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. "
        qs = qs + "Your response should be a valid json file in the following format: \n{\n "
        qs = qs + "\"current number\": \"x\", \n"
        qs = qs + "\"target number\": \"x\", \n"
        qs = qs + "\"thoughts\": \"{first read out the current and target number, then think carefully about which action to choose}\", \n"
        qs = qs + "\"action\": \"-\" or \"+\" \n}"
    elif env_name == 'gym_cards/Blackjack-v0':
        qs = "You are a blackjack player. You are observing the current game state, you can choose between ['stand', 'hit']. "
        qs = qs + "Your response should be a valid json file in the following format: \n {\n "
        qs = qs + "\"thoughts\": \"{first describe your total points and the dealer's total points then think about which action to choose}\", \n"
        qs = qs + "\"action\": \"stand\" or \"hit\" \n}"

    elif env_name == 'gym_cards/EZPoints-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert card game player. You are observing two cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 12, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: "
        qs = qs + "\{\n"
        qs = qs + " \"cards\": [x, y], \n"
        qs = qs + f"\"current formula\": {text_formula}, \n"
        qs = qs + "\"thoughts\": {First check whether the current formula 'z' is complete. "
        qs = qs + "If the current formula 'z' is complete, output '='. "
        qs = qs + "Otherwise consider which number or operator should be appended to the current formula to make it equal 12.} \n"
        qs = qs + "\"action\": \"{number}\" or \"{operator}\" \n \}"

    elif env_name == 'gym_cards/Points24-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert 24 points card game player. You are observing these four cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 24, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: "
        qs = qs + "\{\n"
        qs = qs + " \"cards\": [x, y, z, w], \n"
        qs = qs + f"\"current formula\": {text_formula}, \n"
        qs = qs + "\"thoughts\": {First check whether the current formula equals 24. "
        qs = qs + "If the current formula equals 24, output '='. "
        qs = qs + "Otherwise consider which number or operator should be appended to the current formula to make it equal 24.} \n"
        qs = qs + "\"action\": \"{number}\" or \"{operator}\" \n \}"
    return qs

def get_action_only_prompt(env_name, infos=None):
    """
        This function defines the "action only" prompt for the text-to-action task, depending on the environments
        env_name: determines the prompts for each environment
        info: additional information that can be added to the prompt, if none, then use the default prompt
    """
    if 'maze' in env_name.lower() or 'gym_maze' in env_name.lower():
        # Maze environment action-only prompt
        qs = (
            "You are navigating a maze. You can see the maze layout in the image. "
            "The green circle represents your current position, and the red circle represents the goal. "
            "Your goal is to choose the next step. "
            "You can choose between the following actions: ['N', 'S', 'E', 'W'], "
            "where N=up, S=down, E=right, W=left. "
            "Your response should be a valid JSON object in the following format:\n"
            "{\n"
            '  "action": "N" or "S" or "E" or "W"\n'
            "}"
        )
        return qs
    elif env_name == 'gym_cards/NumberLine-v0':
        qs = "You are playing a game called number line. You will see a target number and a current number in the image. "
        qs = qs + "And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. "
        qs = qs + "Your response should be a valid json file in the following format: \n{\n "
        qs = qs + "\"action\": \"-\" or \"+\" \n}"

    elif env_name == 'gym_cards/Blackjack-v0':
        qs = "You are a blackjack player. You are observing the current game state, you can choose between ['stand', 'hit']. "
        qs = qs + "Your response should be a valid json file in the following format: \n {\n "
        qs = qs + "\"action\": \"stand\" or \"hit\" \n}"

    elif env_name == 'gym_cards/EZPoints-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert card game player. You are observing two cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 12, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: "
        qs = qs + "\{\n\"action\": \"{number}\" or \"{operator}\" \n \}"

    elif env_name == 'gym_cards/Points24-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert 24 points card game player. You are observing these four cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 24, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: "
        qs = qs + "\{\n\"action\": \"{number}\" or \"{operator}\" \n \}"
    return qs

# Define the function that processes the list of strings according to the specified rules
def text_projection(text_actions: List[str], env_name):
    output_indices = []
    if 'maze' in env_name.lower() or 'gym_maze' in env_name.lower():
        action_list = ["n", "s", "e", "w"]
    elif env_name == 'gym_cards/NumberLine-v0':
        action_list = ["-", "+"]
    elif env_name == 'gym_cards/Blackjack-v0':
        action_list = ["stand", "hit"]
    elif env_name == 'gym_cards/EZPoints-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    elif env_name == 'gym_cards/Points24-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    else:
        raise NotImplementedError("Action list not implemented for this env!")
    for string in text_actions:
        if not isinstance(string, str):
            output_indices.append(random.randint(0, len(action_list) - 1))
            continue
        string = string.lower()
        action_index = string.find('"action":')
        if action_index != -1:
            string = string[action_index:]
        contained_actions = []
        if 'points' in env_name.lower() and '10' in string:
            contained_actions.append('10')
            string = string.replace('10', '')  # Remove '10' to prevent it from being counted as '1'
        for action in action_list:
            if action in string:
                contained_actions.append(action)
        contained_actions = list(set(contained_actions))
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            output_indices.append(action_list.index(contained_actions[0]))
        else:
            output_indices.append(random.randint(0, len(action_list) - 1))
    return torch.Tensor([output_indices]).long().reshape(-1, 1)
def text_projection_multi_actions(text_actions: List[str], env_name):
    """
    Parse multiple actions from JSON format with "actions" array for each process.
    Returns a list of action sequences, where each sequence is a list of action indices.
    
    Args:
        text_actions: List of strings, one per process (num_processes)
        env_name: Environment name to determine action mapping
    
    Returns:
        List[List[int]]: List of action sequences, one per process
        Each inner list contains action indices in order
    """
    # Environment expects these action indices: [0="n", 1="s", 2="e", 3="w"]
    action_list_map = {
        'maze': ["n", "s", "e", "w"],
        'gym_maze': ["n", "s", "e", "w"],
    }
    
    # Mapping from model output words to environment action letters
    # Model outputs: "up", "down", "left", "right"
    # Environment expects: "n", "s", "w", "e"
    direction_mapping = {
        'up': 'n',
        'down': 's',
        'left': 'w',
        'right': 'e',
        # Also handle single letters in case model outputs them
        'n': 'n', 'north': 'n',
        's': 's', 'south': 's',
        'e': 'e', 'east': 'e',
        'w': 'w', 'west': 'w',
    }
    
    # Get action list for this environment
    action_list = None
    for key in action_list_map:
        if key in env_name.lower():
            action_list = action_list_map[key]
            break
    
    if action_list is None:
        raise NotImplementedError(f"Action list not implemented for env: {env_name}")
    
    output_actions = []
    
    for string in text_actions:
        process_actions = []
        
        # Skip if not a string
        if not isinstance(string, str):
            # Return empty sequence or single random action
            process_actions.append(random.randint(0, len(action_list) - 1))
            output_actions.append(process_actions)
            continue
        
        # Try to parse as JSON
        actions_array = []
        try:
            # First, try to find and extract JSON object from the string
            # Look for opening brace and try to find matching closing brace
            brace_start = string.find('{')
            if brace_start != -1:
                # Find matching closing brace by counting braces
                brace_count = 0
                brace_end = -1
                for i in range(brace_start, len(string)):
                    if string[i] == '{':
                        brace_count += 1
                    elif string[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = i
                            break
                
                if brace_end != -1:
                    json_str = string[brace_start:brace_end + 1]
                    try:
                        parsed = json.loads(json_str)
                        # Extract actions array
                        if isinstance(parsed, dict) and "actions" in parsed:
                            actions_array = parsed["actions"]
                        elif isinstance(parsed, dict) and "action" in parsed:
                            # Fallback: single action instead of array
                            actions_array = [parsed["action"]]
                    except json.JSONDecodeError:
                        # JSON might be malformed, try fallback parsing
                        pass
                else:
                    # JSON is incomplete (no closing brace found)
                    # Try to extract actions array from incomplete JSON
                    actions_match = re.search(r'"actions"\s*:\s*\[(.*?)(?:\]|$)', string[brace_start:], re.IGNORECASE | re.DOTALL)
                    if actions_match:
                        array_content = actions_match.group(1)
                        # Extract quoted strings even if array is incomplete
                        quoted_strings = re.findall(r'["\']([^"\']+)["\']', array_content)
                        if quoted_strings:
                            actions_array = quoted_strings
                
                # If JSON parsing failed, try parsing the whole string
                if len(actions_array) == 0:
                    try:
                        parsed = json.loads(string)
                        if isinstance(parsed, dict) and "actions" in parsed:
                            actions_array = parsed["actions"]
                        elif isinstance(parsed, dict) and "action" in parsed:
                            actions_array = [parsed["action"]]
                    except json.JSONDecodeError:
                        pass
                
        except Exception as e:
            # If JSON parsing completely fails, try regex fallback
            pass
        
        # Fallback: Try to extract actions array using regex if JSON parsing failed
        if len(actions_array) == 0:
            # Look for array pattern like ["N", "S", "E"] or ['N', 'S', 'E']
            array_match = re.search(r'\[(.*?)\]', string, re.DOTALL)
            if array_match:
                array_content = array_match.group(1)
                # Extract quoted strings from the array
                quoted_strings = re.findall(r'["\']([^"\']+)["\']', array_content)
                if quoted_strings:
                    actions_array = quoted_strings
                else:
                    # Try without quotes - look for comma-separated values
                    # Check if it's in the "actions" field
                    actions_match = re.search(r'"actions"\s*:\s*\[(.*?)\]', string, re.IGNORECASE | re.DOTALL)
                    if actions_match:
                        array_content = actions_match.group(1)
                        quoted_strings = re.findall(r'["\']([^"\']+)["\']', array_content)
                        actions_array = quoted_strings
        
        for action_str in actions_array:
            if not isinstance(action_str, str):
                action_str = str(action_str)
            
            action_str = action_str.lower().strip()
            
            # Map model output ("up", "down", "left", "right") to environment actions ("n", "s", "w", "e")
            if action_str in direction_mapping:
                action_str = direction_mapping[action_str]
            
            if action_str in action_list:
                process_actions.append(action_list.index(action_str))
            else:
                # Invalid action - use random action silently
                process_actions.append(random.randint(0, len(action_list) - 1))
        
        if len(process_actions) == 0:
            # No valid actions found, use random action
            process_actions.append(random.randint(0, len(action_list) - 1))
        
        output_actions.append(process_actions)    
    return output_actions

def grpo_maze_parse(text_actions: List[str], env_name: str = None):
    if env_name is None:
        env_name = "maze"
    return text_projection_multi_actions(text_actions, env_name)
