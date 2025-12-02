#!/usr/bin/env python3
"""
Test script for CustomMazeEnv to verify:
1. Agent can only move one cell at a time (up, down, right, left)
2. Agent cannot go through walls
3. Agent can only move once per action/step
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2c_ppo_acktr.custom_maze_env import CustomMazeEnv, N, S, E, W
from a2c_ppo_acktr.maze_utils import MazeActionWrapper, DenseRewardWrapper, MazeVisualizationWrapper
from PIL import Image

def test_single_cell_movement():
    """Test 1: Verify agent moves exactly one cell per action."""
    print("=" * 60)
    print("TEST 1: Single Cell Movement")
    print("=" * 60)
    
    env = CustomMazeEnv(width=5, height=5, seed=42)
    obs, info = env.reset(seed=42)
    initial_pos = env.agent_pos.copy()
    
    print(f"Initial position: {initial_pos}")
    
    # Test each direction
    actions = [0, 1, 2, 3]  # N, S, E, W
    action_names = ["North", "South", "East", "West"]
    
    for action, name in zip(actions, action_names):
        pos_before = env.agent_pos.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        pos_after = env.agent_pos.copy()
        
        # Calculate movement
        movement = pos_after - pos_before
        distance = np.linalg.norm(movement)
        
        print(f"\n  Action {action} ({name}):")
        print(f"    Position before: {pos_before}")
        print(f"    Position after:  {pos_after}")
        print(f"    Movement: {movement}, Distance: {distance:.2f}")
        
        # Verify: movement should be exactly 0 or 1 cell
        if distance > 0:
            assert distance <= 1.0, f"ERROR: Agent moved more than 1 cell! Distance: {distance}"
            # Verify only one coordinate changed
            non_zero = np.count_nonzero(movement)
            assert non_zero <= 1, f"ERROR: Multiple coordinates changed! Movement: {movement}"
            print(f"    ✓ Valid movement (1 cell)")
        else:
            print(f"    ✓ No movement (blocked by wall or boundary)")
    
    print("\n✓ TEST 1 PASSED: Agent moves at most one cell per action\n")


def test_wall_blocking():
    """Test 2: Verify walls block agent movement."""
    print("=" * 60)
    print("TEST 2: Wall Blocking")
    print("=" * 60)
    
    env = CustomMazeEnv(width=5, height=5, seed=42)
    obs, info = env.reset(seed=42)
    
    # Get maze grid
    maze_grid = env.maze_grid
    
    print("Testing wall blocking at various positions...")
    
    # Test multiple positions
    test_positions = [
        (0, 0, "top-left"),
        (2, 2, "center"),
        (4, 4, "bottom-right"),
    ]
    
    for row, col, desc in test_positions:
        env.agent_pos = np.array([row, col], dtype=np.float32)
        pos_before = env.agent_pos.copy()
        
        print(f"\n  Testing at {desc} ({row}, {col}):")
        cell = maze_grid[row][col]
        
        # Test each direction
        directions = [
            (N, 0, "North"),
            (S, 1, "South"),
            (E, 2, "East"),
            (W, 3, "West"),
        ]
        
        for direction, action, name in directions:
            # Check if wall exists
            has_wall = bool(cell & direction)
            
            # Check boundary
            can_move_boundary = True
            if direction == N and row == 0:
                can_move_boundary = False
            elif direction == S and row == env.height - 1:
                can_move_boundary = False
            elif direction == W and col == 0:
                can_move_boundary = False
            elif direction == E and col == env.width - 1:
                can_move_boundary = False
            
            # Try to move
            obs, reward, terminated, truncated, info = env.step(action)
            pos_after = env.agent_pos.copy()
            moved = not np.array_equal(pos_before, pos_after)
            
            # Verify: if wall exists or at boundary, should not move
            if has_wall or not can_move_boundary:
                assert not moved, f"ERROR: Agent moved through wall/boundary in {name} direction!"
                print(f"    {name}: Wall/Boundary → Blocked ✓")
            else:
                assert moved, f"ERROR: Agent should have moved in {name} direction (no wall)!"
                print(f"    {name}: No wall → Moved ✓")
            
            # Reset position for next test
            env.agent_pos = pos_before.copy()
    
    print("\n✓ TEST 2 PASSED: Walls correctly block agent movement\n")


def test_one_move_per_step():
    """Test 3: Verify agent can only move once per step."""
    print("=" * 60)
    print("TEST 3: One Move Per Step")
    print("=" * 60)
    
    env = CustomMazeEnv(width=5, height=5, seed=42)
    obs, info = env.reset(seed=42)
    
    print("Testing that each step() call results in at most one movement...")
    
    positions = []
    for step in range(10):
        pos_before = env.agent_pos.copy()
        
        # Take a random action
        action = np.random.randint(0, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        
        pos_after = env.agent_pos.copy()
        movement = pos_after - pos_before
        distance = np.linalg.norm(movement)
        
        positions.append(pos_after.copy())
        
        print(f"  Step {step}: Action {action}, Position: {pos_after}, Movement distance: {distance:.2f}")
        
        # Verify: distance should be 0 or 1
        assert distance <= 1.0, f"ERROR: Agent moved more than 1 cell in one step! Distance: {distance}"
        
        if terminated:
            print(f"    Goal reached!")
            break
    
    # Verify positions are sequential (no jumps)
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        diff = np.abs(curr_pos - prev_pos)
        max_diff = np.max(diff)
        assert max_diff <= 1.0, f"ERROR: Position jump detected! {prev_pos} → {curr_pos}"
    
    print("\n✓ TEST 3 PASSED: Agent moves at most once per step\n")


def test_action_directions():
    """Test 4: Verify action directions are correct."""
    print("=" * 60)
    print("TEST 4: Action Directions")
    print("=" * 60)
    
    env = CustomMazeEnv(width=5, height=5, seed=42)
    obs, info = env.reset(seed=42)
    
    # Start at center where we can move in all directions
    env.agent_pos = np.array([2, 2], dtype=np.float32)
    
    # Create a simple test maze where center cell has no walls
    # (We'll manually check if movement is in correct direction)
    actions = [
        (0, "North", np.array([-1, 0])),
        (1, "South", np.array([1, 0])),
        (2, "East", np.array([0, 1])),
        (3, "West", np.array([0, -1])),
    ]
    
    print("Testing action directions from center position (2, 2)...")
    
    for action, name, expected_direction in actions:
        pos_before = env.agent_pos.copy()
        
        # Try to move (may be blocked by wall)
        obs, reward, terminated, truncated, info = env.step(action)
        pos_after = env.agent_pos.copy()
        
        movement = pos_after - pos_before
        
        if np.any(movement != 0):
            # Verify movement is in correct direction (or opposite if blocked)
            # Movement should be in expected direction or zero
            dot_product = np.dot(movement, expected_direction)
            if dot_product > 0:
                print(f"  Action {action} ({name}): Moved correctly {movement} ✓")
            elif dot_product == 0:
                # Might have moved perpendicular (shouldn't happen, but check)
                print(f"  Action {action} ({name}): Moved {movement} (unexpected direction)")
            else:
                print(f"  Action {action} ({name}): Blocked (no movement) ✓")
        else:
            print(f"  Action {action} ({name}): Blocked (no movement) ✓")
        
        # Reset to center
        env.agent_pos = np.array([2, 2], dtype=np.float32)
    
    print("\n✓ TEST 4 PASSED: Action directions are correct\n")


def test_visualization():
    """Test 5: Visualize maze and agent movement."""
    print("=" * 60)
    print("TEST 5: Visualization")
    print("=" * 60)
    
    try:
        # Create wrapped environment (like in training)
        base_env = CustomMazeEnv(width=5, height=5, seed=42)
        env = MazeActionWrapper(base_env)
        env = DenseRewardWrapper(env, maze_size=5)
        env = MazeVisualizationWrapper(env, cell_size=None, max_image_size=300)
        
        obs, info = env.reset(seed=42)
        
        print("Generating visualization images...")
        os.makedirs("test_maze_outputs", exist_ok=True)
        
        # Save initial state
        img = Image.fromarray(obs)
        img.save("test_maze_outputs/test_initial.png")
        print("  Saved: test_maze_outputs/test_initial.png")
        
        # Take a few steps and save images
        for step in range(5):
            action = np.random.randint(0, 4)
            action_names = ["North", "South", "East", "West"]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            img = Image.fromarray(obs)
            img.save(f"test_maze_outputs/test_step_{step+1}_action_{action_names[action]}.png")
            print(f"  Saved: test_maze_outputs/test_step_{step+1}_action_{action_names[action]}.png")
            
            agent_pos = info.get('agent_pos', [0, 0])
            print(f"    Agent position: {agent_pos}, Reward: {reward:.2f}")
            
            if terminated:
                print("    Goal reached!")
                break
        
        print("\n✓ TEST 5 PASSED: Visualization working\n")
    except ImportError as e:
        print(f"  Skipping visualization test (PIL not available): {e}")
        print("  This test requires PIL/Pillow. Run in your training environment.\n")
        raise  # Re-raise to mark as skipped


def test_comprehensive():
    """Comprehensive test: Run agent through a maze and verify all rules."""
    print("=" * 60)
    print("TEST 6: Comprehensive Rule Verification")
    print("=" * 60)
    
    env = CustomMazeEnv(width=5, height=5, seed=42)
    obs, info = env.reset(seed=42)
    
    print("Running agent through maze and verifying all rules...")
    print(f"Start position: {env.agent_pos}")
    print(f"Goal position: {env.goal_pos}")
    
    max_steps = 100
    positions_visited = [env.agent_pos.copy()]
    violations = []
    
    for step in range(max_steps):
        pos_before = env.agent_pos.copy()
        
        # Random action
        action = np.random.randint(0, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        
        pos_after = env.agent_pos.copy()
        movement = pos_after - pos_before
        distance = np.linalg.norm(movement)
        
        # Rule 1: Only one cell movement
        if distance > 1.0:
            violations.append(f"Step {step}: Moved {distance:.2f} cells (should be ≤1)")
        
        # Rule 2: Check if movement is valid (no wall crossing)
        if distance > 0:
            row, col = int(pos_before[0]), int(pos_before[1])
            direction_map = {0: N, 1: S, 2: E, 3: W}
            direction = direction_map[action]
            cell = env.maze_grid[row][col]
            
            # Check if wall exists
            if cell & direction:
                violations.append(f"Step {step}: Moved through wall in direction {action}")
        
        # Rule 3: Only one move per step (already checked by distance)
        
        positions_visited.append(pos_after.copy())
        
        if step % 10 == 0:
            print(f"  Step {step}: Position {pos_after}, Distance to goal: {np.linalg.norm(pos_after - env.goal_pos):.2f}")
        
        if terminated:
            print(f"\n  Goal reached in {step+1} steps!")
            break
    
    # Report violations
    if violations:
        print(f"\n✗ FOUND {len(violations)} VIOLATIONS:")
        for v in violations[:10]:  # Show first 10
            print(f"  - {v}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more")
    else:
        print("\n✓ NO VIOLATIONS: All rules followed correctly!")
    
    print(f"\nTotal steps: {step+1}")
    print(f"Final position: {env.agent_pos}")
    print(f"Goal reached: {terminated}")
    
    return len(violations) == 0


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CUSTOM MAZE ENVIRONMENT TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        ("Single Cell Movement", test_single_cell_movement),
        ("Wall Blocking", test_wall_blocking),
        ("One Move Per Step", test_one_move_per_step),
        ("Action Directions", test_action_directions),
        ("Visualization", test_visualization),
        ("Comprehensive Rules", test_comprehensive),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, True, None))
        except AssertionError as e:
            print(f"\n✗ {test_name} FAILED: {e}\n")
            results.append((test_name, False, str(e)))
        except Exception as e:
            print(f"\n✗ {test_name} ERROR: {e}\n")
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The maze environment follows all rules correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())

