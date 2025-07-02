#!/usr/bin/env python3
"""GPU 자동 감지 및 설정 스크립트"""

import torch
import sys

def detect_available_gpus():
    """사용 가능한 GPU 감지"""
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs detected.")
        return []
    
    num_gpus = torch.cuda.device_count()
    print(f"Total GPUs detected: {num_gpus}")
    
    gpu_info = []
    for i in range(num_gpus):
        try:
            name = torch.cuda.get_device_name(i)
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            gpu_info.append({
                'id': i,
                'name': name,
                'memory_gb': round(mem_total, 2)
            })
            print(f"GPU {i}: {name} ({mem_total:.2f} GB)")
        except Exception as e:
            print(f"Error accessing GPU {i}: {e}")
    
    return gpu_info

def suggest_gpu_config(gpu_info):
    """GPU 설정 제안"""
    if not gpu_info:
        return []
    
    # 메모리가 큰 GPU부터 선택
    sorted_gpus = sorted(gpu_info, key=lambda x: x['memory_gb'], reverse=True)
    
    # 상위 2개 GPU 선택 (또는 가용한 모든 GPU)
    suggested = [gpu['id'] for gpu in sorted_gpus[:min(2, len(sorted_gpus))]]
    
    print(f"\nSuggested GPU configuration: {suggested}")
    return suggested

if __name__ == "__main__":
    print("=== GPU Detection Tool ===")
    gpu_info = detect_available_gpus()
    
    if gpu_info:
        suggested = suggest_gpu_config(gpu_info)
        print(f"\nTo use these GPUs, run:")
        print(f"python main_parallel.py --gpu_ids {' '.join(map(str, suggested))}")
    else:
        print("\nNo GPUs available. Will use CPU.")
        sys.exit(1)