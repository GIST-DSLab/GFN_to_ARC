#!/usr/bin/env python3
"""
ARC Trajectory Transformer Full Experiment Runner (YAML Config Version)
Complete pipeline: Data preprocessing -> Training -> Inference -> Evaluation
"""

import os
import sys
import argparse
import yaml
import subprocess
import time
from datetime import datetime
import logging
from pathlib import Path


class ARCTrajectoryExperiment:
    """Orchestrates the complete trajectory transformer experiment"""
    
    def __init__(self, config_path: str = "configs/config.yaml", experiment_name: str = None):
        # Load configuration
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.experiment_name = experiment_name or f"arc_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        self.experiment_dir = os.path.join(self.config['results_dir'], self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info(f"Starting ARC Trajectory Transformer Experiment: {self.experiment_name}")
        self.logger.info(f"Configuration: {config_path}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def setup_logging(self):
        """Setup experiment logging"""
        log_file = os.path.join(self.experiment_dir, 'experiment.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_command(self, command: str, description: str, capture_output: bool = True) -> bool:
        """Run a command and log the results"""
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Starting: {description}")
        self.logger.info(f"Command: {command}")
        self.logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            if capture_output:
                # 출력을 캡처하는 경우 (데이터 전처리, 추론 등)
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    self.logger.info(f"SUCCESS: {description} (took {duration:.1f}s)")
                    if result.stdout:
                        self.logger.info(f"Output: {result.stdout}")
                    return True
                else:
                    self.logger.error(f"FAILED: {description} (took {duration:.1f}s)")
                    self.logger.error(f"Error: {result.stderr}")
                    return False
            else:
                # 실시간 출력이 필요한 경우 (학습 등)
                self.logger.info(f"🚀 Running command with live output...")
                result = subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                
                duration = time.time() - start_time
                self.logger.info(f"SUCCESS: {description} (took {duration:.1f}s)")
                return True
                
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            self.logger.error(f"FAILED: {description} (took {duration:.1f}s)")
            self.logger.error(f"Exit code: {e.returncode}")
            return False
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"EXCEPTION: {description} (took {duration:.1f}s)")
            self.logger.error(f"Exception: {str(e)}")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check if all required files and directories exist"""
        self.logger.info("Checking prerequisites...")
        
        required_files = [
            "data_preprocessing.py",
            "training.py", 
            "inference.py",
            "models/arc_transformer.py",
            "utils/data_utils.py"
        ]
        
        required_dirs = [
            self.config['trajectory_data_dir'],
            self.config['rearc_data_dir']
        ]
        
        all_good = True
        
        # Check files
        for file in required_files:
            if not os.path.exists(file):
                self.logger.error(f"Missing required file: {file}")
                all_good = False
            else:
                self.logger.info(f"✓ Found: {file}")
        
        # Check directories
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                self.logger.error(f"Missing required directory: {dir_path}")
                all_good = False
            else:
                self.logger.info(f"✓ Found: {dir_path}")
        
        # Create output directories
        for dir_key in ['processed_data_dir', 'model_save_dir', 'results_dir']:
            dir_path = self.config[dir_key]
            os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"✓ Created/verified: {dir_path}")
        
        # Check for trajectory data
        if os.path.exists(self.config['trajectory_data_dir']):
            trajectory_dirs = [d for d in os.listdir(self.config['trajectory_data_dir']) 
                             if d.startswith('problem_')]
            self.logger.info(f"Found {len(trajectory_dirs)} trajectory directories")
        
        return all_good
    
    def run_data_preprocessing(self, skip_if_exists: bool = True) -> bool:
        """Run data preprocessing step"""
        processed_file = os.path.join(self.config['processed_data_dir'], "train_trajectories.pt")
        
        if skip_if_exists and os.path.exists(processed_file):
            self.logger.info("Preprocessed data already exists, skipping...")
            return True
        
        command = f"python data_preprocessing.py --config {self.config_path}"
        return self.run_command(command, "Data Preprocessing")
    
    def run_training(self, skip_if_exists: bool = True, gpu_id: int = None) -> bool:
        """Run model training"""
        model_file = os.path.join(self.config['model_save_dir'], "arc_transformer_best.pt")
        
        if skip_if_exists and os.path.exists(model_file):
            self.logger.info("Trained model already exists, skipping...")
            return True
        
        command = f"python training.py --config {self.config_path}"
        if gpu_id is not None:
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
        
        return self.run_command(command, "Model Training", capture_output=False)
    
    def run_inference_and_evaluation(self, gpu_id: int = None) -> bool:
        """Run inference and evaluation"""
        command = f"python inference.py --config {self.config_path} --output_dir {self.experiment_dir}"
        if gpu_id is not None:
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
        
        return self.run_command(command, "Inference and Evaluation")
    
    def generate_experiment_report(self):
        """Generate final experiment report"""
        report_file = os.path.join(self.experiment_dir, "experiment_report.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"ARC Trajectory Transformer Experiment Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Experiment Name: {self.experiment_name}\n")
            f.write(f"Configuration: {self.config_path}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"\n")
            
            # Check for results
            results_file = os.path.join(self.experiment_dir, "evaluation_results.json")
            if os.path.exists(results_file):
                import json
                with open(results_file, 'r') as rf:
                    results = json.load(rf)
                
                f.write(f"Overall Accuracy: {results.get('overall_accuracy', 'N/A'):.2%}\n")
                f.write(f"Total Tests: {results.get('total_tests', 'N/A')}\n")
                f.write(f"Total Correct: {results.get('total_correct', 'N/A')}\n")
                f.write(f"\n")
                
                if 'problem_results' in results:
                    f.write(f"Problem-wise Results:\n")
                    for problem_id, problem_data in results['problem_results'].items():
                        f.write(f"  Problem {problem_id}: {problem_data['accuracy']:.2%} "
                               f"({problem_data['correct_count']}/{problem_data['total_count']})\n")
            else:
                f.write(f"No evaluation results found.\n")
        
        self.logger.info(f"Experiment report saved to: {report_file}")
    
    def run_full_experiment(self, skip_existing: bool = True, gpu_id: int = None) -> bool:
        """Run the complete experiment pipeline"""
        self.logger.info("Starting full experiment pipeline...")
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            self.logger.error("Prerequisites check failed!")
            return False
        
        # Step 2: Data preprocessing
        self.logger.info("\n📊 Step 1/3: Data Preprocessing")
        if not self.run_data_preprocessing(skip_if_exists=skip_existing):
            self.logger.error("Data preprocessing failed!")
            return False
        
        # Step 3: Training
        self.logger.info("\n🎯 Step 2/3: Model Training")
        if not self.run_training(skip_if_exists=skip_existing, gpu_id=gpu_id):
            self.logger.error("Model training failed!")
            return False
        
        # Step 4: Inference and Evaluation
        self.logger.info("\n🔮 Step 3/3: Inference and Evaluation")
        if not self.run_inference_and_evaluation(gpu_id=gpu_id):
            self.logger.error("Inference and evaluation failed!")
            return False
        
        # Step 5: Generate report
        self.generate_experiment_report()
        
        self.logger.info("\n✅ Experiment completed successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(description='Run ARC Trajectory Transformer Experiment')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use (default: use CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run all steps even if outputs exist')
    parser.add_argument('--preprocessing_only', action='store_true',
                       help='Run only data preprocessing')
    parser.add_argument('--training_only', action='store_true',
                       help='Run only model training')
    parser.add_argument('--inference_only', action='store_true',
                       help='Run only inference and evaluation')
    
    args = parser.parse_args()
    
    # Create experiment instance
    experiment = ARCTrajectoryExperiment(
        config_path=args.config,
        experiment_name=args.experiment_name
    )
    
    try:
        if args.preprocessing_only:
            success = experiment.run_data_preprocessing(skip_if_exists=not args.force)
        elif args.training_only:
            success = experiment.run_training(skip_if_exists=not args.force, gpu_id=args.gpu)
        elif args.inference_only:
            success = experiment.run_inference_and_evaluation(gpu_id=args.gpu)
        else:
            success = experiment.run_full_experiment(skip_existing=not args.force, gpu_id=args.gpu)
        
        if success:
            print(f"\n✅ Experiment completed successfully!")
            print(f"📁 Results available in: {experiment.experiment_dir}")
        else:
            print(f"\n❌ Experiment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()