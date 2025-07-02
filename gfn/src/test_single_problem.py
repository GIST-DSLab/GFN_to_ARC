#!/usr/bin/env python3
"""
Test script to verify the parallel training implementation works for a single problem.
This helps ensure the system works before running full parallel training.
"""

import torch
import argparse
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_parallel import train_single_problem

def test_single_problem():
    """Test training on a single problem with small number of trajectories."""
    
    # Create a mock args object
    class MockArgs:
        def __init__(self):
            self.batch_size = 1
            self.num_epochs = 1
            self.env_mode = "entire"
            self.num_actions = 5
            self.ep_len = 10
            self.device = 0
            self.use_offpolicy = False
            self.sampling_method = "prt"
            self.save_trajectories = True
            self.num_trajectories = 100  # Small number for testing
            self.subtask_num = 2
            self.output_dir = "test_output"
            self.checkpoint_interval = 50
    
    args = MockArgs()
    
    # Create test output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test arguments for problem 53
    test_args = {
        'prob_index': 53,
        'args': args,
        'process_id': 0
    }
    
    print("Starting test training on problem 53...")
    print(f"Output will be saved to: {args.output_dir}")
    
    try:
        train_single_problem(test_args)
        print("\nTest completed successfully!")
        
        # Check if output files were created
        problem_dir = os.path.join(args.output_dir, "problem_53")
        if os.path.exists(problem_dir):
            files = os.listdir(problem_dir)
            print(f"\nGenerated files in {problem_dir}:")
            for f in files:
                print(f"  - {f}")
        else:
            print(f"\nWarning: Expected output directory {problem_dir} was not created")
            
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_problem()