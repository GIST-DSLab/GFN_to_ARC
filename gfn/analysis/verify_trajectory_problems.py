#!/usr/bin/env python3
"""
Verify that trajectory data corresponds to the correct ARC tasks by comparing grids
"""
import json
import os
import numpy as np

def load_arc_task(task_id):
    """Load ARC task data from the original dataset."""
    arc_dirs = [
        "/home/ubuntu/GFN_to_ARC/gfn/src/LLM_experiment/data/re-arc/arc_original/training",
        "/home/ubuntu/GFN_to_ARC/gfn/src/LLM_experiment/data/re-arc/arc_original/evaluation"
    ]
    
    for arc_dir in arc_dirs:
        task_file = os.path.join(arc_dir, f"{task_id}.json")
        if os.path.exists(task_file):
            with open(task_file, 'r') as f:
                return json.load(f)
    return None

def load_trajectory_sample(problem_id):
    """Load a sample trajectory from the GFN data."""
    trajectory_dir = f"/data/gflownet-llm/problem_{problem_id}"
    trajectory_file = os.path.join(trajectory_dir, "trajectories_0_1000.json")
    
    if os.path.exists(trajectory_file):
        try:
            with open(trajectory_file, 'r') as f:
                data = json.load(f)
                return data[0] if data else None  # Get first trajectory
        except Exception as e:
            print(f"Error loading trajectory for problem {problem_id}: {e}")
    return None

def compare_grids(arc_grid, trajectory_grid, description=""):
    """Compare two grids and return similarity information."""
    if not isinstance(arc_grid, list) or not isinstance(trajectory_grid, list):
        return False, "Grid format mismatch"
    
    if len(arc_grid) != len(trajectory_grid):
        return False, f"Height mismatch: ARC={len(arc_grid)}, Trajectory={len(trajectory_grid)}"
    
    if not arc_grid or len(arc_grid[0]) != len(trajectory_grid[0]):
        return False, f"Width mismatch: ARC={len(arc_grid[0]) if arc_grid else 0}, Trajectory={len(trajectory_grid[0])}"
    
    # Convert to numpy arrays for comparison
    arc_array = np.array(arc_grid)
    traj_array = np.array(trajectory_grid)
    
    if arc_array.shape != traj_array.shape:
        return False, f"Shape mismatch: ARC={arc_array.shape}, Trajectory={traj_array.shape}"
    
    # Check if grids are identical
    if np.array_equal(arc_array, traj_array):
        return True, "Grids are identical"
    
    # Calculate similarity percentage
    total_cells = arc_array.size
    matching_cells = np.sum(arc_array == traj_array)
    similarity = (matching_cells / total_cells) * 100
    
    return False, f"Grids differ: {similarity:.1f}% similarity ({matching_cells}/{total_cells} cells match)"

def verify_problem_mapping():
    """Verify that trajectory data matches the expected ARC tasks."""
    problem_mappings = {
        86: "3c9b0459",
        139: "6150a2bd", 
        149: "67a3c6ac",
        154: "68b16354",
        178: "74dd1130",
        240: "9dfd6313",
        379: "ed36ccf7"
    }
    
    print("🔍 Verifying GFN Trajectory Data vs ARC Tasks")
    print("=" * 80)
    
    for problem_id, task_id in problem_mappings.items():
        print(f"\n📋 Problem {problem_id} -> ARC Task {task_id}")
        print("-" * 50)
        
        # Load ARC task data
        arc_task = load_arc_task(task_id)
        if not arc_task:
            print(f"❌ Could not load ARC task {task_id}")
            continue
        
        # Load trajectory sample
        trajectory = load_trajectory_sample(problem_id)
        if not trajectory:
            print(f"❌ Could not load trajectory for problem {problem_id}")
            continue
        
        print(f"✅ Successfully loaded both ARC task and trajectory data")
        
        # Show ARC task info
        train_examples = arc_task.get('train', [])
        test_examples = arc_task.get('test', [])
        print(f"📊 ARC Task Info: {len(train_examples)} training examples, {len(test_examples)} test examples")
        
        # Compare input grids from training examples with trajectory initial state
        if train_examples and 'states_full' in trajectory and trajectory['states_full']:
            print(f"\n🔍 Comparing grids...")
            
            # Get first training example input
            arc_input = train_examples[0]['input']
            print(f"ARC Input Grid Shape: {len(arc_input)}x{len(arc_input[0]) if arc_input else 0}")
            print(f"ARC Input Grid:")
            for row in arc_input[:5]:  # Show first 5 rows
                print(f"  {row}")
            if len(arc_input) > 5:
                print(f"  ... ({len(arc_input)} total rows)")
            
            # Get trajectory initial state
            traj_state = trajectory['states_full'][0]
            if 'input' in traj_state:
                traj_input_raw = traj_state['input']
                traj_input_dim = traj_state.get('input_dim', [3, 3])  # Default to 3x3
                
                # Check if it's already a 2D grid structure or needs reshaping
                if isinstance(traj_input_raw, list) and len(traj_input_raw) > 0 and isinstance(traj_input_raw[0], list):
                    # Already a 2D structure, extract the relevant portion
                    traj_input = []
                    for i in range(min(traj_input_dim[0], len(traj_input_raw))):
                        row = traj_input_raw[i][:traj_input_dim[1]]
                        traj_input.append(row)
                elif len(traj_input_raw) == 900:  # 30x30 flattened
                    # Convert flat 30x30 to 2D, then extract the relevant portion
                    full_grid = []
                    for i in range(30):
                        row = traj_input_raw[i*30:(i+1)*30]
                        full_grid.append(row)
                    
                    # Extract only the relevant portion based on input_dim
                    traj_input = []
                    for i in range(traj_input_dim[0]):
                        row = full_grid[i][:traj_input_dim[1]]
                        traj_input.append(row)
                else:
                    # Flat list that needs reshaping based on input_dim
                    traj_input = []
                    idx = 0
                    for i in range(traj_input_dim[0]):
                        row = []
                        for j in range(traj_input_dim[1]):
                            if idx < len(traj_input_raw):
                                row.append(traj_input_raw[idx])
                            idx += 1
                        traj_input.append(row)
                
                print(f"\nTrajectory Input Grid Shape: {len(traj_input)}x{len(traj_input[0]) if traj_input else 0}")
                print(f"Trajectory Input Grid:")
                for row in traj_input[:5]:  # Show first 5 rows
                    print(f"  {row}")
                if len(traj_input) > 5:
                    print(f"  ... ({len(traj_input)} total rows)")
                
                # Compare grids
                match, details = compare_grids(arc_input, traj_input, "Input")
                if match:
                    print(f"✅ {details}")
                else:
                    print(f"⚠️  {details}")
                    
                    # If grids don't match, check if they could be different training examples
                    print(f"\n🔄 Checking other training examples...")
                    found_match = False
                    for i, example in enumerate(train_examples[1:3], 1):  # Check next 2 examples
                        match, details = compare_grids(example['input'], traj_input, f"Training example {i}")
                        if match:
                            print(f"✅ Found match with training example {i}: {details}")
                            found_match = True
                            break
                        else:
                            print(f"   Example {i}: {details}")
                    
                    if not found_match:
                        print(f"❌ No matching training examples found")
            
            # Also check if trajectory contains expected problem_id
            if 'problem_id' in trajectory:
                traj_problem_id = trajectory['problem_id']
                if traj_problem_id == problem_id:
                    print(f"✅ Trajectory problem_id ({traj_problem_id}) matches expected ({problem_id})")
                else:
                    print(f"❌ Trajectory problem_id ({traj_problem_id}) doesn't match expected ({problem_id})")
            
            # Show trajectory stats
            if 'actions' in trajectory and 'rewards' in trajectory:
                actions = trajectory['actions']
                rewards = trajectory['rewards']
                max_reward = max(rewards) if rewards else 0
                print(f"📈 Trajectory Stats: {len(actions)} actions, max reward: {max_reward:.2f}")
                
        else:
            print(f"❌ Cannot compare grids - missing data")

if __name__ == "__main__":
    verify_problem_mapping()