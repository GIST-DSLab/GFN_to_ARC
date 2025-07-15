#!/usr/bin/env python3
"""
Analyze all successful trajectories for all 7 problems and create plots
"""
import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_trajectory_files(problem_id):
    """Load all trajectory files for a given problem."""
    data_dir = f"/data/gflownet-llm/problem_{problem_id}"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found for problem {problem_id}: {data_dir}")
        return []
    
    all_trajectories = []
    batch_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    batch_files.sort()  # Process in order
    
    print(f"📂 Loading {len(batch_files)} files for problem {problem_id}")
    
    for i, filename in enumerate(batch_files):
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                batch_data = json.load(f)
                all_trajectories.extend(batch_data)
                
            if (i + 1) % 10 == 0:  # Progress every 10 files
                print(f"  Loaded {i + 1}/{len(batch_files)} files...")
                
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_trajectories)} total trajectories for problem {problem_id}")
    return all_trajectories

def extract_action_sequence(trajectory):
    """Extract action sequence from trajectory."""
    if 'actions' not in trajectory:
        return None
    
    actions = trajectory['actions']
    # Convert to string representation for uniqueness check
    action_str = ','.join([str(action) for action in actions])
    return action_str

def analyze_problem_trajectories(problem_id, max_trajectories=1000000000):
    """Analyze trajectories for a single problem."""
    print(f"\n🔍 Analyzing Problem {problem_id}")
    print("=" * 50)
    
    trajectories = load_trajectory_files(problem_id)
    
    if max_trajectories and len(trajectories) > max_trajectories:
        trajectories = trajectories[:max_trajectories]
        print(f"📊 Limited analysis to first {max_trajectories} trajectories for performance")
    
    # Separate successful and unsuccessful trajectories
    successful_trajectories = []
    total_trajectories = len(trajectories)
    
    print(f"🔄 Processing {total_trajectories} trajectories...")
    for i, traj in enumerate(trajectories):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i:,}/{total_trajectories:,} trajectories...")
        
        # Check if any state has is_correct = 1
        is_successful = False
        if 'states_full' in traj and len(traj['states_full']) > 0:
            for state in traj['states_full']:
                if state.get('is_correct', 0) == 1:
                    is_successful = True
                    break
        
        if is_successful:
            successful_trajectories.append(traj)
    
    success_count = len(successful_trajectories)
    success_rate = (success_count / total_trajectories * 100) if total_trajectories > 0 else 0
    
    print(f"📈 Total trajectories: {total_trajectories}")
    print(f"✅ Successful trajectories: {success_count} ({success_rate:.1f}%)")
    
    if success_count == 0:
        return {
            'problem_id': problem_id,
            'total_count': total_trajectories,
            'successful_count': 0,
            'success_rate': 0.0,
            'unique_count': 0,
            'unique_rate': 0.0,
            'cumulative_data': {'trajectories': [], 'unique_counts': []}
        }
    
    # Analyze uniqueness among successful trajectories
    unique_sequences = set()
    cumulative_unique = []
    cumulative_trajectories = []
    
    print(f"🔄 Analyzing uniqueness among {success_count} successful trajectories...")
    
    batch_size = 500  # Process in larger batches for cumulative analysis
    for i in range(0, success_count, batch_size):
        batch = successful_trajectories[i:i+batch_size]
        
        for traj in batch:
            action_seq = extract_action_sequence(traj)
            if action_seq:
                unique_sequences.add(action_seq)
        
        # Record cumulative progress
        cumulative_trajectories.append(i + len(batch))
        cumulative_unique.append(len(unique_sequences))
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch):,}/{success_count:,} successful trajectories...")
    
    unique_count = len(unique_sequences)
    unique_rate = (unique_count / success_count * 100) if success_count > 0 else 0
    
    print(f"🎯 Unique successful sequences: {unique_count} ({unique_rate:.1f}%)")
    
    return {
        'problem_id': problem_id,
        'total_count': total_trajectories,
        'successful_count': success_count,
        'success_rate': success_rate,
        'unique_count': unique_count,
        'unique_rate': unique_rate,
        'cumulative_data': {
            'trajectories': cumulative_trajectories,
            'unique_counts': cumulative_unique
        }
    }

def create_comprehensive_plots(results):
    """Create all 4 plots based on analysis results."""
    output_dir = "/home/ubuntu/GFN_to_ARC/gfn/analysis"
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Filter results with successful trajectories for plots 1, 2, 4
    valid_results = [r for r in results if r['successful_count'] > 0]
    
    # Plot 1: Cumulative Unique Sequence Growth
    plt.figure(figsize=(14, 10))
    for i, result in enumerate(valid_results):
        color = colors[i % len(colors)]
        problem_id = result['problem_id']
        
        x = result['cumulative_data']['trajectories']
        y = result['cumulative_data']['unique_counts']
        
        if len(x) > 0 and len(y) > 0:
            plt.plot(x, y, label=f'Problem {problem_id + 1}', 
                    color=color, marker='o', linewidth=3, markersize=6)
    
    plt.xlabel('Cumulative Successful Trajectories', fontsize=48, fontweight='bold')
    plt.ylabel('Cumulative Unique Action Sequences', fontsize=48, fontweight='bold')
    plt.title('Growth of Unique Action Sequences in Successful Trajectories', fontsize=54, fontweight='bold')
    plt.legend(fontsize=42)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=42)
    plt.yticks(fontsize=42)
    plt.tight_layout()
    
    plot1_path = os.path.join(output_dir, "final_plot1_cumulative_unique.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 1 saved: {plot1_path}")
    
    # Plot 2: New Unique Sequences Discovered Per Batch
    plt.figure(figsize=(14, 10))
    for i, result in enumerate(valid_results):
        color = colors[i % len(colors)]
        problem_id = result['problem_id']
        
        unique_counts = result['cumulative_data']['unique_counts']
        if len(unique_counts) > 1:
            # Calculate new discoveries per batch
            new_discoveries = [unique_counts[0]] + [unique_counts[j] - unique_counts[j-1] 
                                                  for j in range(1, len(unique_counts))]
            x = result['cumulative_data']['trajectories']
            
            plt.plot(x, new_discoveries, label=f'Problem {problem_id + 1}', 
                    color=color, marker='s', linewidth=3, markersize=6)
    
    plt.xlabel('Cumulative Successful Trajectories', fontsize=48, fontweight='bold')
    plt.ylabel('New Unique Sequences Discovered', fontsize=48, fontweight='bold')
    plt.title('Rate of New Unique Sequence Discovery', fontsize=54, fontweight='bold')
    plt.legend(fontsize=42)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=42)
    plt.yticks(fontsize=42)
    plt.tight_layout()
    
    plot2_path = os.path.join(output_dir, "final_plot2_new_discoveries.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 2 saved: {plot2_path}")
    
    # Plot 3: Success Rate Comparison (all problems)
    plt.figure(figsize=(14, 10))
    problem_ids = [r['problem_id'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    
    bars = plt.bar(range(len(problem_ids)), success_rates,
                   color=[colors[i % len(colors)] for i in range(len(problem_ids))],
                   alpha=0.7, edgecolor='black', linewidth=1, width=0.5)
    
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + max(success_rates)*0.02,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=42)
        else:
            plt.text(bar.get_x() + bar.get_width()/2., 2,
                    '0%', ha='center', va='bottom', fontweight='bold', fontsize=42)
    
    plt.xlabel('Problem ID', fontsize=48, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=48, fontweight='bold')
    plt.title('Trajectory Success Rate by Problem', fontsize=54, fontweight='bold')
    plt.xticks(range(len(problem_ids)), [f'P{pid + 1}' for pid in problem_ids], fontsize=42)
    plt.yticks(fontsize=42)
    plt.ylim(0, max(success_rates) * 1.15 if success_rates else 50)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot3_path = os.path.join(output_dir, "final_plot3_success_rate.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 3 saved: {plot3_path}")
    
    # Plot 4: Final Uniqueness Rate (only successful problems)
    plt.figure(figsize=(14, 10))
    valid_problem_ids = [r['problem_id'] for r in valid_results]
    unique_rates = [r['unique_rate'] for r in valid_results]
    
    if len(valid_results) > 0:
        bars = plt.bar(range(len(valid_problem_ids)), unique_rates,
                       color=[colors[i % len(colors)] for i in range(len(valid_problem_ids))],
                       alpha=0.7, edgecolor='black', linewidth=1, width=0.5)
        
        for i, (bar, rate) in enumerate(zip(bars, unique_rates)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(unique_rates)*0.02,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=42)
        
        plt.ylim(0, max(unique_rates) * 1.15)
    
    plt.xlabel('Problem ID', fontsize=48, fontweight='bold')
    plt.ylabel('Uniqueness Rate Among Successful (%)', fontsize=48, fontweight='bold')
    plt.title('Final Uniqueness Rate Among Successful Trajectories', fontsize=54, fontweight='bold')
    plt.xticks(range(len(valid_problem_ids)), [f'P{pid + 1}' for pid in valid_problem_ids], fontsize=42)
    plt.yticks(fontsize=42)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot4_path = os.path.join(output_dir, "final_plot4_uniqueness_rate.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 4 saved: {plot4_path}")
    
    return [plot1_path, plot2_path, plot3_path, plot4_path]

def main():
    """Main analysis function."""
    print("🚀 Starting comprehensive successful trajectory analysis for all 7 problems")
    print("=" * 80)
    
    problem_ids = [86, 139, 149, 154, 178, 240, 379]
    results = []
    
    # Analyze each problem
    for problem_id in problem_ids:
        try:
            result = analyze_problem_trajectories(problem_id)
            results.append(result)
        except Exception as e:
            print(f"❌ Error analyzing problem {problem_id}: {e}")
            # Add empty result to maintain order
            results.append({
                'problem_id': problem_id,
                'total_count': 0,
                'successful_count': 0,
                'success_rate': 0.0,
                'unique_count': 0,
                'unique_rate': 0.0,
                'cumulative_data': {'trajectories': [], 'unique_counts': []}
            })
    
    # Create plots
    print(f"\n📊 Creating comprehensive plots...")
    plot_paths = create_comprehensive_plots(results)
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"📋 FINAL ANALYSIS SUMMARY")
    print(f"=" * 80)
    
    for result in results:
        pid = result['problem_id']
        success_rate = result['success_rate']
        unique_rate = result['unique_rate']
        successful_count = result['successful_count']
        unique_count = result['unique_count']
        
        print(f"Problem {pid:3d}: {successful_count:5d} successful ({success_rate:5.1f}%) | "
              f"{unique_count:5d} unique ({unique_rate:5.1f}%)")
    
    print(f"\n✅ Analysis complete! Created 4 plots based on actual 100k trajectory data.")
    
    return results, plot_paths

if __name__ == "__main__":
    main()