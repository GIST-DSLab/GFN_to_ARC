#!/usr/bin/env python3
"""
ARC Trajectory Transformer Full Experiment Runner
Complete pipeline: Data preprocessing -> Training -> Inference -> Evaluation
"""

import os
import sys
import argparse
import json
import subprocess
import time
from datetime import datetime
import logging
from tqdm import tqdm


class ARCTrajectoryExperiment:
    """Orchestrates the complete trajectory transformer experiment"""
    
    def __init__(self, config_name: str = "base", experiment_name: str = None):
        self.config_name = config_name
        self.experiment_name = experiment_name or f"arc_transformer_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        self.experiment_dir = os.path.join("./experiments", self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info(f"Starting ARC Trajectory Transformer Experiment: {self.experiment_name}")
        self.logger.info(f"Configuration: {config_name}")
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
    
    def run_command(self, command: str, description: str) -> bool:
        """Run a shell command and log the result"""
        self.logger.info(f"Starting: {description}")
        self.logger.info(f"Command: {command}")
        
        start_time = time.time()
        
        try:
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
            "utils/data_utils.py",
            "configs/arc_config.py"
        ]
        
        required_dirs = [
            "../LLM_experiment/data/trajectories_output",
            "../LLM_experiment/data/re-arc"
        ]
        
        all_good = True
        
        # Check files
        for file_path in required_files:
            if not os.path.exists(file_path):
                self.logger.error(f"Required file not found: {file_path}")
                all_good = False
            else:
                self.logger.info(f"✓ Found: {file_path}")
        
        # Check directories
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                self.logger.error(f"Required directory not found: {dir_path}")
                all_good = False
            else:
                self.logger.info(f"✓ Found: {dir_path}")
        
        return all_good
    
    def run_preprocessing(self) -> bool:
        """Run data preprocessing"""
        self.logger.info("=== STEP 1: Data Preprocessing ===")
        
        command = f"python data_preprocessing.py --config_name {self.config_name} --analyze"
        return self.run_command(command, "Data preprocessing")
    
    def run_training(self, use_wandb: bool = False) -> bool:
        """Run model training"""
        self.logger.info("=== STEP 2: Model Training ===")
        
        # Update config to use experiment directory
        experiment_model_dir = os.path.join(self.experiment_dir, "models")
        experiment_results_dir = os.path.join(self.experiment_dir, "results")
        
        command = f"python training.py --config {self.config_name}"
        if use_wandb:
            command += " --wandb"
        
        # Set environment variables for custom paths
        env_vars = f"PYTHONPATH={os.getcwd()}:$PYTHONPATH "
        command = env_vars + command
        
        success = self.run_command(command, "Model training")
        
        if success:
            # Copy trained models to experiment directory
            os.makedirs(experiment_model_dir, exist_ok=True)
            if os.path.exists("./models"):
                copy_cmd = f"cp -r ./models/* {experiment_model_dir}/"
                subprocess.run(copy_cmd, shell=True)
                self.logger.info(f"Models copied to {experiment_model_dir}")
        
        return success
    
    def run_inference(self) -> bool:
        """Run model inference and evaluation"""
        self.logger.info("=== STEP 3: Model Inference & Evaluation ===")
        
        # Find best model
        model_dir = os.path.join(self.experiment_dir, "models")
        if not os.path.exists(model_dir):
            model_dir = "./models"
        
        best_model_path = os.path.join(model_dir, "checkpoint_best.pt")
        if not os.path.exists(best_model_path):
            self.logger.warning("Best model not found, using latest checkpoint")
            best_model_path = os.path.join(model_dir, "checkpoint_latest.pt")
        
        if not os.path.exists(best_model_path):
            self.logger.error("No trained model found!")
            return False
        
        # Run inference
        output_dir = os.path.join(self.experiment_dir, "evaluation")
        command = f"python inference.py --config {self.config_name} --model_path {best_model_path} --output_dir {output_dir}"
        
        return self.run_command(command, "Model inference and evaluation")
    
    def generate_experiment_report(self) -> bool:
        """Generate experiment summary report"""
        self.logger.info("=== STEP 4: Generating Experiment Report ===")
        
        try:
            report = {
                "experiment_name": self.experiment_name,
                "config": self.config_name,
                "timestamp": datetime.now().isoformat(),
                "experiment_dir": self.experiment_dir
            }
            
            # Load training stats if available
            training_stats_path = os.path.join(self.experiment_dir, "results", "training_stats.json")
            if os.path.exists(training_stats_path):
                with open(training_stats_path, 'r') as f:
                    training_stats = json.load(f)
                report["training_stats"] = training_stats
            
            # Load evaluation results if available
            eval_results_path = os.path.join(self.experiment_dir, "evaluation", "evaluation_results.json")
            if os.path.exists(eval_results_path):
                with open(eval_results_path, 'r') as f:
                    eval_results = json.load(f)
                report["evaluation_results"] = eval_results
                
                # Extract key metrics
                report["summary"] = {
                    "overall_accuracy": eval_results.get("overall_accuracy", 0.0),
                    "total_correct": eval_results.get("total_correct", 0),
                    "total_tests": eval_results.get("total_tests", 0),
                    "problems_evaluated": len(eval_results.get("problem_results", []))
                }
            
            # Save report
            report_path = os.path.join(self.experiment_dir, "experiment_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Print summary
            self.logger.info("\n" + "="*60)
            self.logger.info(f"EXPERIMENT COMPLETED: {self.experiment_name}")
            self.logger.info("="*60)
            
            if "summary" in report:
                summary = report["summary"]
                self.logger.info(f"Overall Accuracy: {summary['overall_accuracy']:.3f}")
                self.logger.info(f"Correct Predictions: {summary['total_correct']}/{summary['total_tests']}")
                self.logger.info(f"Problems Evaluated: {summary['problems_evaluated']}")
            
            if "training_stats" in report:
                training_stats = report["training_stats"]
                if training_stats.get("best_val_loss"):
                    self.logger.info(f"Best Validation Loss: {training_stats['best_val_loss']:.4f}")
            
            self.logger.info(f"Experiment Report: {report_path}")
            self.logger.info("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate experiment report: {e}")
            return False
    
    def run_full_experiment(self, skip_preprocessing: bool = False, 
                           skip_training: bool = False, 
                           skip_inference: bool = False,
                           use_wandb: bool = False) -> bool:
        """Run the complete experiment pipeline"""
        
        start_time = time.time()
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.logger.error("Prerequisites check failed. Aborting experiment.")
            return False
        
        success = True
        
        # Step 1: Data preprocessing
        if not skip_preprocessing:
            if not self.run_preprocessing():
                self.logger.error("Data preprocessing failed. Aborting experiment.")
                return False
        else:
            self.logger.info("Skipping data preprocessing")
        
        # Step 2: Training
        if not skip_training:
            if not self.run_training(use_wandb=use_wandb):
                self.logger.error("Training failed. Aborting experiment.")
                return False
        else:
            self.logger.info("Skipping training")
        
        # Step 3: Inference
        if not skip_inference:
            if not self.run_inference():
                self.logger.error("Inference failed.")
                success = False
        else:
            self.logger.info("Skipping inference")
        
        # Step 4: Report generation
        self.generate_experiment_report()
        
        # Final summary
        total_time = time.time() - start_time
        self.logger.info(f"Total experiment time: {total_time/60:.1f} minutes")
        
        return success


def main():
    parser = argparse.ArgumentParser(description="Run ARC Trajectory Transformer Experiment")
    parser.add_argument("--config", type=str, default="base",
                       choices=["base", "small"], help="Configuration name")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name (default: auto-generated)")
    parser.add_argument("--skip_preprocessing", action="store_true",
                       help="Skip data preprocessing step")
    parser.add_argument("--skip_training", action="store_true", 
                       help="Skip training step")
    parser.add_argument("--skip_inference", action="store_true",
                       help="Skip inference step")
    parser.add_argument("--wandb", action="store_true",
                       help="Use Weights & Biases for training")
    parser.add_argument("--preprocessing_only", action="store_true",
                       help="Only run data preprocessing")
    parser.add_argument("--training_only", action="store_true",
                       help="Only run training")
    parser.add_argument("--inference_only", action="store_true",
                       help="Only run inference")
    
    args = parser.parse_args()
    
    # Create experiment runner
    experiment = ARCTrajectoryExperiment(
        config_name=args.config,
        experiment_name=args.name
    )
    
    try:
        if args.preprocessing_only:
            success = experiment.run_preprocessing()
        elif args.training_only:
            success = experiment.run_training(use_wandb=args.wandb)
        elif args.inference_only:
            success = experiment.run_inference()
        else:
            # Run full experiment
            success = experiment.run_full_experiment(
                skip_preprocessing=args.skip_preprocessing,
                skip_training=args.skip_training,
                skip_inference=args.skip_inference,
                use_wandb=args.wandb
            )
        
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