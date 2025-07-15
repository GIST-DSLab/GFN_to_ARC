#!/usr/bin/env python3
"""
Quick check for reward vs correctness discrepancies
"""

import json
import os

def quick_check_single_problem(problem_id):
    """Quick check for a single problem using first trajectory file."""
    data_dir = f"/data/gflownet-llm/problem_{problem_id}"
    
    if not os.path.exists(data_dir):
        return None
    
    # Load just the first trajectory file
    trajectory_file = os.path.join(data_dir, "trajectories_0_1000.json")
    if not os.path.exists(trajectory_file):
        return None
    
    try:
        with open(trajectory_file, 'r') as f:
            trajectories = json.load(f)
    except:
        return None
    
    # Analyze first 100 trajectories
    results = {
        'total': 0,
        'reward_pos_correct_true': 0,
        'reward_pos_correct_false': 0,  # DISCREPANCY 1
        'reward_zero_correct_true': 0,  # DISCREPANCY 2  
        'reward_zero_correct_false': 0,
        'examples': []
    }
    
    for i, traj in enumerate(trajectories[:100]):
        if not isinstance(traj, dict):
            continue
            
        # Get final reward
        rewards = traj.get('rewards', [])
        if not rewards:
            continue
            
        final_reward = rewards[-1]
        
        # Check is_correct in states
        states_full = traj.get('states_full', [])
        if not states_full:
            continue
            
        has_correct = any(state.get('is_correct', 0) == 1 for state in states_full)
        
        results['total'] += 1
        
        if final_reward > 0 and has_correct:
            results['reward_pos_correct_true'] += 1
        elif final_reward > 0 and not has_correct:
            results['reward_pos_correct_false'] += 1
            # Store example
            if len(results['examples']) < 3:
                results['examples'].append({
                    'type': 'reward_pos_no_correct',
                    'traj_id': traj.get('trajectory_id', i),
                    'final_reward': final_reward,
                    'actions': traj.get('actions', []),
                    'is_correct_values': [s.get('is_correct', 0) for s in states_full]
                })
        elif final_reward <= 0 and has_correct:
            results['reward_zero_correct_true'] += 1
            # Store example
            if len(results['examples']) < 5:
                results['examples'].append({
                    'type': 'reward_zero_has_correct', 
                    'traj_id': traj.get('trajectory_id', i),
                    'final_reward': final_reward,
                    'actions': traj.get('actions', []),
                    'is_correct_values': [s.get('is_correct', 0) for s in states_full]
                })
        else:
            results['reward_zero_correct_false'] += 1
    
    return results

def main():
    print("🔍 Quick Reward vs Correctness Check")
    print("=" * 50)
    
    problem_ids = [86, 139, 149, 154, 178, 240, 379]
    
    for problem_id in problem_ids:
        print(f"\nProblem {problem_id}:")
        result = quick_check_single_problem(problem_id)
        
        if result is None:
            print("  ❌ Could not analyze")
            continue
            
        total = result['total']
        disc1 = result['reward_pos_correct_false'] 
        disc2 = result['reward_zero_correct_true']
        
        print(f"  Total analyzed: {total}")
        print(f"  Reward>0 but no correct: {disc1} ({disc1/total*100:.1f}%)" if total > 0 else f"  Reward>0 but no correct: {disc1}")
        print(f"  Reward=0 but has correct: {disc2} ({disc2/total*100:.1f}%)" if total > 0 else f"  Reward=0 but has correct: {disc2}")
        
        # Show examples
        for ex in result['examples']:
            if ex['type'] == 'reward_pos_no_correct':
                print(f"    🔴 Traj {ex['traj_id']}: reward={ex['final_reward']:.2f}, is_correct={ex['is_correct_values']}")
            elif ex['type'] == 'reward_zero_has_correct':
                print(f"    🟡 Traj {ex['traj_id']}: reward={ex['final_reward']:.2f}, is_correct={ex['is_correct_values']}")

if __name__ == "__main__":
    main()