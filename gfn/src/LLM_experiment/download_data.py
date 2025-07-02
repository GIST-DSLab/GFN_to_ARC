#!/usr/bin/env python3
"""
Download re-arc dataset for LLM experiment
"""

import os
import subprocess
import sys
import zipfile
import json
from pathlib import Path

def run_command(cmd, cwd=None):
    """Execute a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            print(f"Error executing: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        print(f"Success: {cmd}")
        return True
    except Exception as e:
        print(f"Exception running command: {e}")
        return False

def download_rearc_dataset():
    """Download re-arc dataset"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("=== Downloading re-arc dataset ===")
    # Clone re-arc repository
    if not (data_dir / "re-arc").exists():
        if run_command("git clone https://github.com/michaelhodel/re-arc.git", cwd="data"):
            print("✓ re-arc repository cloned successfully")
        else:
            print("✗ Failed to clone re-arc repository")
            return False
    else:
        print("✓ re-arc directory already exists")
    
    # Extract arc_original.zip if it exists
    arc_original_zip = data_dir / "re-arc" / "arc_original.zip"
    if arc_original_zip.exists():
        print("\n=== Extracting arc_original.zip ===")
        try:
            with zipfile.ZipFile(arc_original_zip, 'r') as zip_ref:
                zip_ref.extractall(data_dir / "re-arc")
            print(f"✓ Extracted arc_original.zip")
        except Exception as e:
            print(f"✗ Failed to extract arc_original.zip: {e}")
    
    # Extract re_arc.zip if it exists
    re_arc_zip = data_dir / "re-arc" / "re_arc.zip"
    if re_arc_zip.exists():
        print("\n=== Extracting re_arc.zip ===")
        try:
            with zipfile.ZipFile(re_arc_zip, 'r') as zip_ref:
                extract_path = data_dir / "re-arc" / "re_arc_extracted"
                zip_ref.extractall(extract_path)
            print(f"✓ Extracted re_arc.zip to {extract_path}")
        except Exception as e:
            print(f"✗ Failed to extract re_arc.zip: {e}")
    
    print("\n=== Dataset Structure ===")
    print("✓ re-arc dataset downloaded to data/re-arc/")
    print("  - arc_original/ contains 400 ARC training tasks")
    print("  - Each task has input/output examples for training and testing")
    
    return True

def create_trajectories_dir():
    """Create trajectories_output directory if it doesn't exist"""
    trajectories_dir = Path("data/trajectories_output")
    trajectories_dir.mkdir(exist_ok=True)
    print(f"✓ Created trajectories directory: {trajectories_dir}")
    return True

def setup_llm_experiment_data():
    """Setup data for LLM experiment"""
    print("=== Setting up LLM Experiment Data ===")
    
    if not download_rearc_dataset():
        return False
    
    if not create_trajectories_dir():
        return False
    
    print("\n✅ LLM Experiment data setup complete!")
    print("\nDirectory structure:")
    print("  data/")
    print("  ├── re-arc/           # ARC dataset")
    print("  │   ├── arc_original/ # Original ARC tasks")
    print("  │   └── ...          # Other re-arc files")
    print("  └── trajectories_output/ # GFlowNet trajectory data")
    print("      └── problem_*/   # Problem-specific trajectories")
    
    return True

if __name__ == "__main__":
    if setup_llm_experiment_data():
        print("\n🎉 Ready to run LLM experiments!")
    else:
        print("\n❌ Failed to setup data")
        sys.exit(1)