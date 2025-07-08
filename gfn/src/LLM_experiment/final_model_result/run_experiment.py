#!/usr/bin/env python3
"""
전체 LLM 실험 실행 스크립트
데이터 전처리 → 학습 → 추론 및 평가를 순차적으로 실행
"""

import os
import sys
import time
import subprocess
import argparse
from utils import *
import logging

def run_command(command: str, description: str, logger, capture_output: bool = False):
    """명령어 실행 및 로깅"""
    logger.info(f"{'='*50}")
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    logger.info(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        if capture_output:
            # 출력을 캡처하는 경우 (데이터 전처리, 추론 등)
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Completed: {description} ({elapsed_time:.2f}s)")
            
            if result.stdout:
                logger.info(f"STDOUT:\\n{result.stdout}")
            if result.stderr:
                logger.warning(f"STDERR:\\n{result.stderr}")
        else:
            # 실시간 출력이 필요한 경우 (학습 등)
            logger.info(f"🚀 Running command with live output...")
            result = subprocess.run(
                command, 
                shell=True, 
                check=True
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Completed: {description} ({elapsed_time:.2f}s)")
            
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Failed: {description} ({elapsed_time:.2f}s)")
        logger.error(f"Exit code: {e.returncode}")
        
        if capture_output:
            logger.error(f"STDOUT:\\n{e.stdout}")
            logger.error(f"STDERR:\\n{e.stderr}")
        
        return False

def check_prerequisites(config: Dict, logger):
    """사전 요구사항 확인"""
    logger.info("Checking prerequisites...")
    
    # trajectory 데이터 확인
    trajectory_dir = config['trajectory_data_dir']
    if not os.path.exists(trajectory_dir):
        logger.error(f"Trajectory data directory not found: {trajectory_dir}")
        return False
    
    # 필요한 trajectory 파일들 확인
    missing_files = []
    for problem_id in config['problem_mapping'].keys():
        trajectory_file = os.path.join(
            trajectory_dir, 
            f"problem_{problem_id}", 
            "trajectories_0_1000.json"
        )
        if not os.path.exists(trajectory_file):
            missing_files.append(trajectory_file)
    
    if missing_files:
        logger.warning(f"Missing trajectory files: {missing_files}")
        logger.warning("Some problems will be skipped during processing")
    
    # ReARC 데이터 확인
    rearc_dir = config['rearc_data_dir']
    if not os.path.exists(rearc_dir):
        logger.error(f"ReARC data directory not found: {rearc_dir}")
        return False
    
    # 필요한 Python 패키지 확인
    required_packages = [
        'torch', 'transformers', 'numpy', 'yaml', 
        'sklearn', 'wandb', 'matplotlib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            logger.error(f"Required package not found: {package}")
            return False
    
    logger.info("✅ All prerequisites checked")
    return True

def setup_directories(config: Dict, logger):
    """필요한 디렉토리 생성"""
    directories = [
        config['processed_data_dir'],
        config['model_save_dir'],
        config["results_dir"],
        "./logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")

def run_preprocessing(config: Dict, logger, force: bool = False, config_path: str = "configs/config.yaml"):
    """데이터 전처리 실행"""
    
    # 이미 처리된 데이터가 있는지 확인
    processed_file = os.path.join(config['processed_data_dir'], "all_training_data.json")
    if os.path.exists(processed_file) and not force:
        logger.info("Preprocessed data already exists. Skipping preprocessing.")
        logger.info("Use --force-preprocessing to regenerate.")
        return True
    
    # all_training_data.json이 없어도 개별 problem 파일들이 있으면 건너뛰기
    import glob
    problem_files = glob.glob(os.path.join(config['processed_data_dir'], "problem_*_processed.json"))
    if problem_files and not force:
        # 빈 파일이 아닌 유효한 데이터가 있는지 확인
        valid_files = []
        for problem_file in problem_files:
            try:
                with open(problem_file, 'r') as f:
                    data = f.read().strip()
                    if data and data != "[]" and len(data) > 10:  # 빈 배열이 아니고 충분한 데이터가 있는 경우
                        valid_files.append(problem_file)
            except Exception:
                continue
        
        if valid_files:
            logger.info(f"Found {len(valid_files)} existing processed data files. Skipping preprocessing.")
            logger.info(f"Files: {[os.path.basename(f) for f in valid_files]}")
            logger.info("Use --force-preprocessing to regenerate.")
            return True
    
    python_path = "/home/ubuntu/miniconda3/envs/gflownet/bin/python"
    command = f"{python_path} data_preprocessing.py --config {config_path}"
    return run_command(command, "Data Preprocessing", logger, capture_output=False)  # tqdm 표시를 위해 False로 변경

def run_training(config: Dict, logger, gpu_ids: str = None, force: bool = False, config_path: str = "configs/config.yaml", use_unsloth: bool = False):
    """모델 학습 실행"""
    
    # Unsloth 사용 시 다른 모델 디렉토리 확인
    if use_unsloth:
        model_dir = os.path.join(config['model_save_dir'], "unsloth_lora_model")
    else:
        model_dir = os.path.join(config['model_save_dir'], f"arc_action_model_{config['model_name'].split('/')[-1]}", "final_model")
    
    if os.path.exists(model_dir) and not force:
        logger.info("Trained model already exists. Skipping training.")
        logger.info("Use --force-training to retrain.")
        return True
    
    # Unsloth 사용 시 단일 GPU 전용
    if use_unsloth:
        logger.info("Using Unsloth for accelerated training (single GPU only)")
        python_path = "/home/ubuntu/miniconda3/envs/gflownet/bin/python"  # gflow-llm 환경 사용
        command = f"{python_path} training_unsloth.py --config {config_path}"
        return run_command(command, "Unsloth Training", logger, capture_output=False)
    
    # GPU 개수에 따라 DDP 또는 일반 python 사용
    if gpu_ids:
        gpu_list = gpu_ids.split()
        num_gpus = len(gpu_list)
        
        if num_gpus > 1:
            # 멀티 GPU: torchrun 사용
            logger.info(f"Using DDP with torchrun on {num_gpus} GPUs: {gpu_ids}")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
            
            # torchrun을 통한 DDP 학습 (gflow-llm 환경 사용)
            python_path = "/home/ubuntu/miniconda3/envs/gflownet/bin/python"
            command = f"{python_path} -m torch.distributed.run --nproc-per-node={num_gpus} --nnodes=1 --standalone training.py --config {config_path}"
            return run_command(command, "Model Training", logger, capture_output=False)
        else:
            # 단일 GPU: 일반 python 사용 (gflow-llm 환경)
            logger.info(f"Using single GPU: {gpu_ids}")
            python_path = "/home/ubuntu/miniconda3/envs/gflownet/bin/python"
            command = f"{python_path} training.py --gpu_ids {gpu_ids} --config {config_path}"
            return run_command(command, "Model Training", logger, capture_output=False)
    else:
        # GPU 지정 없음: 일반 python 사용 (gflow-llm 환경)
        python_path = "/home/ubuntu/miniconda3/envs/gflownet/bin/python"
        command = f"{python_path} training.py --config {config_path}"
        return run_command(command, "Model Training", logger, capture_output=False)

def run_inference(config: Dict, logger, gpu_ids: str = None, force: bool = False, config_path: str = "configs/config.yaml"):
    """추론 및 평가 실행"""
    
    # 이미 평가 결과가 있는지 확인
    results_file = os.path.join(config['results_dir'], "evaluation_results.json")
    if os.path.exists(results_file) and not force:
        logger.info("Evaluation results already exist. Skipping inference.")
        logger.info("Use --force-inference to re-evaluate.")
        return True
    
    python_path = "/home/ubuntu/miniconda3/envs/gflownet/bin/python"
    command = f"{python_path} inference.py --config {config_path}"
    if gpu_ids:
        command += f" --gpu_ids {gpu_ids}"
    return run_command(command, "Inference and Evaluation", logger)

def generate_summary_report(config: Dict, logger):
    """실험 결과 요약 보고서 생성"""
    try:
        results_file = os.path.join(config['results_dir'], "evaluation_results.json")
        if not os.path.exists(results_file):
            logger.warning("No evaluation results found for summary report")
            return
        
        results = load_json(results_file)
        
        # 요약 보고서 생성
        summary = {
            "experiment_summary": {
                "model_name": config['model_name'],
                "problems_trained": list(config['problem_mapping'].keys()),
                "overall_accuracy": results.get('overall_stats', {}).get('overall_accuracy', 0),
                "total_tests": results.get('overall_stats', {}).get('total_tests', 0),
                "total_correct": results.get('overall_stats', {}).get('total_correct', 0)
            },
            "problem_breakdown": {}
        }
        
        # 문제별 세부 결과
        for arc_id, problem_result in results.get('problem_results', {}).items():
            summary["problem_breakdown"][arc_id] = {
                "accuracy": problem_result.get('accuracy', 0),
                "correct_count": problem_result.get('correct_count', 0),
                "total_count": problem_result.get('total_count', 0)
            }
        
        # 요약 보고서 저장
        summary_file = os.path.join(config['results_dir'], "experiment_summary.json")
        save_json(summary, summary_file)
        
        # 콘솔에 요약 출력
        logger.info("\\n" + "="*60)
        logger.info("EXPERIMENT SUMMARY REPORT")
        logger.info("="*60)
        logger.info(f"Model: {config['model_name']}")
        logger.info(f"Problems Trained: {len(config['problem_mapping'])}")
        logger.info(f"Overall Accuracy: {summary['experiment_summary']['overall_accuracy']:.3f}")
        logger.info(f"Total Tests: {summary['experiment_summary']['total_tests']}")
        logger.info(f"Total Correct: {summary['experiment_summary']['total_correct']}")
        logger.info("\\nProblem Breakdown:")
        
        for arc_id, stats in summary["problem_breakdown"].items():
            logger.info(f"  {arc_id}: {stats['accuracy']:.3f} ({stats['correct_count']}/{stats['total_count']})")
        
        logger.info("="*60)
        logger.info(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run complete LLM experiment for ARC action sequence learning")
    parser.add_argument("--config", default="configs/config.yaml", help="Configuration file path")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=None, help="Specific GPU IDs to use (e.g., --gpu_ids 6 7)")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip data preprocessing step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference and evaluation step")
    parser.add_argument("--force-preprocessing", action="store_true", help="Force rerun preprocessing even if data exists")
    parser.add_argument("--force-training", action="store_true", help="Force retrain model even if exists")
    parser.add_argument("--force-inference", action="store_true", help="Force re-evaluate even if results exist")
    parser.add_argument("--preprocessing-only", action="store_true", help="Run only preprocessing step")
    parser.add_argument("--training-only", action="store_true", help="Run only training step")
    parser.add_argument("--inference-only", action="store_true", help="Run only inference step")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth for accelerated training (single GPU only)")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    log_dir = config.get('environment', {}).get('log_dir', './logs')
    log_file = os.path.join(log_dir, "experiment.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logging(log_file)
    
    logger.info("🚀 Starting LLM Experiment for ARC Action Sequence Learning")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model: {config['model_name']}")
    
    start_time = time.time()
    
    try:
        # 사전 요구사항 확인
        if not check_prerequisites(config, logger):
            logger.error("Prerequisites check failed. Exiting.")
            return 1
        
        # 디렉토리 설정
        setup_directories(config, logger)
        
        # 단계별 실행
        success = True
        
        # 전처리 단계
        if not args.skip_preprocessing and (not args.training_only and not args.inference_only):
            if not run_preprocessing(config, logger, args.force_preprocessing, args.config):
                success = False
                logger.error("Preprocessing failed")
                
        if args.preprocessing_only:
            logger.info("Preprocessing completed. Exiting as requested.")
            return 0
        
        # GPU IDs 문자열 변환
        gpu_ids_str = " ".join(map(str, args.gpu_ids)) if args.gpu_ids else None
        
        # 학습 단계
        if success and not args.skip_training and (not args.preprocessing_only and not args.inference_only):
            if not run_training(config, logger, gpu_ids_str, args.force_training, args.config, args.use_unsloth):
                success = False
                logger.error("Training failed")
                
        if args.training_only:
            logger.info("Training completed. Exiting as requested.")
            return 0
        
        # 추론 및 평가 단계
        if success and not args.skip_inference and not args.preprocessing_only:
            if not run_inference(config, logger, gpu_ids_str, args.force_inference, args.config):
                success = False
                logger.error("Inference failed")
        
        # 요약 보고서 생성
        if success:
            generate_summary_report(config, logger)
        
        # 최종 결과
        total_time = time.time() - start_time
        
        if success:
            logger.info(f"🎉 Experiment completed successfully! Total time: {total_time:.2f}s ({total_time/60:.1f}m)")
            return 0
        else:
            logger.error(f"❌ Experiment failed. Total time: {total_time:.2f}s")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)