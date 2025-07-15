#!/bin/bash

# Full analysis script for running in tmux
echo "🚀 Starting full reward vs correctness analysis for all 100k trajectories"
echo "Time started: $(date)"
echo "========================================================================================"

cd /home/ubuntu/GFN_to_ARC/gfn/analysis

# Run the analysis
python analyze_reward_vs_correctness.py > full_analysis_output.log 2>&1

echo "========================================================================================"
echo "Analysis completed at: $(date)"
echo "Results saved to:"
echo "  - full_analysis_output.log (console output)"
echo "  - reward_vs_correctness_analysis.json (detailed results)"