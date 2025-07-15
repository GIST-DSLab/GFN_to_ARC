#!/usr/bin/env python3
"""
Test analysis with single file to understand data structure and performance
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_single_file(problem_id):
    """Analyze one file to understand the data structure."""
    file_path = f"/data/gflownet-llm/problem_{problem_id}/trajectories_0_1000.json"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print(f"📂 Loading: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} trajectories")
        
        # Analyze success rate
        successful = 0
        for traj in data:
            if 'rewards' in traj and len(traj['rewards']) > 0:
                if max(traj['rewards']) > 0:
                    successful += 1
        
        success_rate = (successful / len(data)) * 100
        print(f"📈 Success rate: {successful}/{len(data)} ({success_rate:.1f}%)")
        
        # Analyze uniqueness among successful
        if successful > 0:
            unique_sequences = set()
            successful_trajectories = []
            
            for traj in data:
                if 'rewards' in traj and len(traj['rewards']) > 0:
                    if max(traj['rewards']) > 0:
                        successful_trajectories.append(traj)
                        if 'actions' in traj:
                            action_str = ','.join([str(action) for action in traj['actions']])
                            unique_sequences.add(action_str)
            
            unique_count = len(unique_sequences)
            unique_rate = (unique_count / successful) * 100
            print(f"🎯 Unique sequences: {unique_count}/{successful} ({unique_rate:.1f}%)")
            
            return {
                'problem_id': problem_id,
                'total': len(data),
                'successful': successful,
                'success_rate': success_rate,
                'unique': unique_count,
                'unique_rate': unique_rate
            }
        else:
            print(f"❌ No successful trajectories found")
            return {
                'problem_id': problem_id,
                'total': len(data),
                'successful': 0,
                'success_rate': 0,
                'unique': 0,
                'unique_rate': 0
            }
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def test_all_problems():
    """Test all 7 problems with single file each."""
    print("🚀 Single File Analysis Test")
    print("=" * 50)
    
    problem_ids = [86, 139, 149, 154, 178, 240, 379]
    results = []
    
    for problem_id in problem_ids:
        print(f"\n🔍 Testing Problem {problem_id}")
        print("-" * 30)
        result = analyze_single_file(problem_id)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"📋 SINGLE FILE TEST SUMMARY")
    print(f"=" * 50)
    
    for result in results:
        pid = result['problem_id']
        success_rate = result['success_rate']
        unique_rate = result['unique_rate']
        successful = result['successful']
        unique = result['unique']
        
        print(f"Problem {pid:3d}: {successful:4d} successful ({success_rate:5.1f}%) | "
              f"{unique:4d} unique ({unique_rate:5.1f}%)")
    
    return results

if __name__ == "__main__":
    test_all_problems()