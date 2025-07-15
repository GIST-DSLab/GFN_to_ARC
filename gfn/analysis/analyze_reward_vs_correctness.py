#!/usr/bin/env python3
"""
Analyze discrepancies between reward[-1] > 0 and is_correct=1 in trajectory data
"""

import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def load_trajectory_files(problem_id):
    """Load all trajectory files for a given problem."""
    data_dir = f"/data/gflownet-llm/problem_{problem_id}"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found for problem {problem_id}: {data_dir}")
        return []
    
    all_trajectories = []
    batch_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f.startswith('trajectories_')]
    batch_files.sort()
    
    print(f"📂 Loading {len(batch_files)} files for problem {problem_id}")
    
    # Load all files for complete analysis
    # max_files = 3
    # batch_files = batch_files[:max_files]
    
    for i, filename in enumerate(batch_files):
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                batch_data = json.load(f)
                all_trajectories.extend(batch_data)
                
            if (i + 1) % 20 == 0 or i == len(batch_files) - 1:
                print(f"  Loaded {i + 1}/{len(batch_files)} files...")
                
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_trajectories)} total trajectories for problem {problem_id}")
    return all_trajectories

def analyze_single_trajectory(trajectory):
    """Analyze a single trajectory for reward vs correctness discrepancy."""
    traj_id = trajectory.get('trajectory_id', 'unknown')
    problem_id = trajectory.get('problem_id', 'unknown')
    
    # Get final reward
    rewards = trajectory.get('rewards', [])
    if not rewards:
        return None
    
    final_reward = rewards[-1]
    
    # Check is_correct values in all states
    states_full = trajectory.get('states_full', [])
    if not states_full:
        return None
    
    is_correct_values = []
    for state in states_full:
        is_correct_values.append(state.get('is_correct', 0))
    
    # Find if any state has is_correct = 1
    has_correct_state = any(val == 1 for val in is_correct_values)
    
    # Check for discrepancy
    reward_positive = final_reward > 0
    
    analysis = {
        'trajectory_id': traj_id,
        'problem_id': problem_id,
        'final_reward': final_reward,
        'reward_positive': reward_positive,
        'has_correct_state': has_correct_state,
        'is_correct_values': is_correct_values,
        'num_states': len(states_full),
        'actions': trajectory.get('actions', []),
        'all_rewards': rewards
    }
    
    return analysis

def analyze_problem_discrepancies(problem_id, max_trajectories=None):
    """Analyze reward vs correctness discrepancies for a single problem."""
    print(f"\n🔍 Analyzing Problem {problem_id}")
    print("=" * 50)
    
    trajectories = load_trajectory_files(problem_id)
    
    if max_trajectories and len(trajectories) > max_trajectories:
        trajectories = trajectories[:max_trajectories]
        print(f"📊 Limited analysis to first {max_trajectories} trajectories")
    
    # Statistics counters
    stats = {
        'total': 0,
        'reward_positive_correct_true': 0,   # reward > 0 AND has correct state
        'reward_positive_correct_false': 0,  # reward > 0 BUT no correct state
        'reward_zero_correct_true': 0,       # reward = 0 BUT has correct state  
        'reward_zero_correct_false': 0,      # reward = 0 AND no correct state
    }
    
    discrepancy_examples = {
        'reward_positive_but_no_correct': [],  # reward > 0 but is_correct never 1
        'reward_zero_but_has_correct': []      # reward = 0 but is_correct = 1 somewhere
    }
    
    print(f"🔄 Processing {len(trajectories)} trajectories...")
    
    for i, traj in enumerate(tqdm(trajectories, desc="Analyzing trajectories")):
        analysis = analyze_single_trajectory(traj)
        if analysis is None:
            continue
            
        stats['total'] += 1
        
        reward_pos = analysis['reward_positive']
        has_correct = analysis['has_correct_state']
        
        # Categorize
        if reward_pos and has_correct:
            stats['reward_positive_correct_true'] += 1
        elif reward_pos and not has_correct:
            stats['reward_positive_correct_false'] += 1
            # This is a discrepancy - store example
            if len(discrepancy_examples['reward_positive_but_no_correct']) < 10:
                discrepancy_examples['reward_positive_but_no_correct'].append(analysis)
        elif not reward_pos and has_correct:
            stats['reward_zero_correct_true'] += 1
            # This is also a discrepancy - store example
            if len(discrepancy_examples['reward_zero_but_has_correct']) < 10:
                discrepancy_examples['reward_zero_but_has_correct'].append(analysis)
        else:  # not reward_pos and not has_correct
            stats['reward_zero_correct_false'] += 1
    
    return stats, discrepancy_examples

def main():
    """Main analysis function."""
    print("🚀 Starting reward vs correctness discrepancy analysis")
    print("=" * 80)
    
    problem_ids = [86, 139, 149, 154, 178, 240, 379]
    all_stats = {}
    all_discrepancies = {}
    
    for problem_id in problem_ids:
        try:
            stats, discrepancies = analyze_problem_discrepancies(problem_id)
            all_stats[problem_id] = stats
            all_discrepancies[problem_id] = discrepancies
        except Exception as e:
            print(f"❌ Error analyzing problem {problem_id}: {e}")
            all_stats[problem_id] = {'total': 0, 'error': str(e)}
            all_discrepancies[problem_id] = {}
    
    # Summary report
    print(f"\n" + "=" * 80)
    print(f"📋 DISCREPANCY ANALYSIS SUMMARY")
    print(f"=" * 80)
    
    total_across_all = 0
    total_discrepancy_1 = 0  # reward > 0 but no correct state
    total_discrepancy_2 = 0  # reward = 0 but has correct state
    
    for problem_id in problem_ids:
        stats = all_stats.get(problem_id, {})
        if 'error' in stats:
            print(f"Problem {problem_id:3d}: ERROR - {stats['error']}")
            continue
            
        total = stats.get('total', 0)
        disc_1 = stats.get('reward_positive_correct_false', 0)
        disc_2 = stats.get('reward_zero_correct_true', 0)
        
        total_across_all += total
        total_discrepancy_1 += disc_1
        total_discrepancy_2 += disc_2
        
        print(f"Problem {problem_id:3d}: {total:6d} total | "
              f"reward>0 but no_correct: {disc_1:4d} | "
              f"reward=0 but has_correct: {disc_2:4d}")
    
    print(f"\n{'='*50}")
    print(f"TOTAL ACROSS ALL PROBLEMS:")
    print(f"  Total trajectories analyzed: {total_across_all:,}")
    if total_across_all > 0:
        print(f"  Reward > 0 but no correct state: {total_discrepancy_1:,} ({total_discrepancy_1/total_across_all*100:.2f}%)")
        print(f"  Reward = 0 but has correct state: {total_discrepancy_2:,} ({total_discrepancy_2/total_across_all*100:.2f}%)")
    else:
        print(f"  Reward > 0 but no correct state: {total_discrepancy_1:,}")
        print(f"  Reward = 0 but has correct state: {total_discrepancy_2:,}")
    
    # Show example discrepancies
    print(f"\n" + "=" * 80)
    print("📝 EXAMPLE DISCREPANCIES")
    print("=" * 80)
    
    for problem_id in problem_ids:
        discrepancies = all_discrepancies.get(problem_id, {})
        
        # Type 1: reward > 0 but no correct state
        examples_1 = discrepancies.get('reward_positive_but_no_correct', [])
        if examples_1:
            print(f"\nProblem {problem_id} - Reward > 0 but no correct state:")
            for i, ex in enumerate(examples_1[:3]):  # Show first 3 examples
                print(f"  Example {i+1}: Traj {ex['trajectory_id']}, "
                      f"Reward: {ex['final_reward']:.2f}, "
                      f"Actions: {ex['actions']}, "
                      f"is_correct values: {ex['is_correct_values']}")
        
        # Type 2: reward = 0 but has correct state  
        examples_2 = discrepancies.get('reward_zero_but_has_correct', [])
        if examples_2:
            print(f"\nProblem {problem_id} - Reward = 0 but has correct state:")
            for i, ex in enumerate(examples_2[:3]):  # Show first 3 examples
                print(f"  Example {i+1}: Traj {ex['trajectory_id']}, "
                      f"Reward: {ex['final_reward']:.2f}, "
                      f"Actions: {ex['actions']}, "
                      f"is_correct values: {ex['is_correct_values']}")
    
    # Save detailed results
    output_file = "/home/ubuntu/GFN_to_ARC/gfn/analysis/reward_vs_correctness_analysis.json"
    results = {
        'summary_stats': all_stats,
        'discrepancy_examples': all_discrepancies,
        'analysis_metadata': {
            'total_problems': len(problem_ids),
            'problem_ids': problem_ids,
            'total_trajectories': total_across_all,
            'total_discrepancy_type_1': total_discrepancy_1,
            'total_discrepancy_type_2': total_discrepancy_2
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Detailed analysis saved to: {output_file}")
    return results

if __name__ == "__main__":
    main()