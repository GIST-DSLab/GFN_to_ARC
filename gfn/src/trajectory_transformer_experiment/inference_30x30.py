#!/usr/bin/env python3
"""
ARC Trajectory Transformer Inference Script for 30x30 grids
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import logging
import yaml

from models.arc_transformer_30x30 import create_30x30_model
from utils.data_utils import create_vocabulary, pad_grid_to_30x30, flatten_30x30_grid

class ARC30x30Inference:
    """30x30 ARC Trajectory Transformer Inference Engine"""
    
    def __init__(self, config, model_path: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = create_vocabulary()
        self.vocab_size = config.get('vocab_size', 26)
        
        # Setup logging
        self.setup_logging()
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Initialize action executor
        self.executor = self.create_executor()
        
        self.logger.info(f"Inference engine initialized on {self.device}")
    
    def create_executor(self):
        """Create action executor for ARC environment"""
        class ActionExecutor:
            def execute_action_sequence(self, initial_grid, actions):
                """Execute sequence of actions on grid"""
                current_grid = [row[:] for row in initial_grid]  # Deep copy
                
                if not actions:
                    return current_grid
                
                for action in actions:
                    rows, cols = len(current_grid), len(current_grid[0])
                    if action == 0:  # left_rotate
                        current_grid = [[current_grid[j][rows-1-i] for j in range(cols)] for i in range(rows)]
                    elif action == 1:  # right_rotate
                        current_grid = [[current_grid[cols-1-j][i] for j in range(cols)] for i in range(rows)]
                    elif action == 2:  # horizontal_flip
                        current_grid = [row[::-1] for row in current_grid]
                    elif action == 3:  # vertical_flip
                        current_grid = current_grid[::-1]
                    elif action == 4:  # submit
                        break
                
                return current_grid
        
        return ActionExecutor()
    
    def setup_logging(self):
        """Setup logging"""
        log_file = os.path.join(self.config['results_dir'], 'inference_30x30.log')
        os.makedirs(self.config['results_dir'], exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with saved config
        model_config = checkpoint.get('config', self.config)
        model = create_30x30_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model loaded successfully")
        self.logger.info(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"Training loss: {checkpoint.get('loss', 'unknown')}")
        
        return model
    
    def encode_grid_30x30(self, grid: List[List[int]]) -> List[int]:
        """Encode grid to 30x30 padded format"""
        # Pad to 30x30
        padded_flat = flatten_30x30_grid(grid)
        
        # Convert to tokens (0-9 for colors, 10 for padding)
        tokens = [int(x) if x <= 10 else 10 for x in padded_flat]
        
        self.logger.info(f"Encoded grid from shape {len(grid)}x{len(grid[0])} to 30x30 (900 tokens)")
        return tokens
    
    def decode_action_sequence(self, generated_tokens: List[int]) -> List[int]:
        """Extract action sequence from generated tokens"""
        actions = []
        
        # Look for action tokens (11-15) in the generated sequence
        for i, token in enumerate(generated_tokens):
            if 11 <= token <= 15:
                action = token - 11
                actions.append(action)
                self.logger.info(f"Found action {action} at position {i}")
                
                if token == 15:  # Submit action
                    break
        
        return actions
    
    def generate_action_sequence(self, initial_grid: List[List[int]], 
                                max_actions: int = 20) -> Tuple[List[int], float]:
        """Generate action sequence for given grid"""
        # Encode grid to 30x30
        grid_tokens = self.encode_grid_30x30(initial_grid)
        
        # Create input tensor
        input_ids = torch.tensor(grid_tokens, device=self.device).unsqueeze(0)
        
        self.logger.info(f"Input sequence length: {input_ids.size(1)}")
        
        # Generate sequence
        with torch.no_grad():
            # We need to generate: action + reward + value tokens
            max_new_tokens = max_actions * 3
            
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=self.config.get('temperature', 1.0),
                top_k=self.config.get('top_k', None),
                top_p=self.config.get('top_p', 0.9),
                pad_token_id=10,
                eos_token_id=15
            )
            
            # Extract generated tokens (excluding input)
            generated_tokens = generated[0, input_ids.size(1):].cpu().tolist()
            
            self.logger.info(f"Generated {len(generated_tokens)} tokens")
            
            # Calculate confidence
            if len(generated_tokens) > 0:
                # Simple confidence based on generation length
                confidence = min(1.0, len(generated_tokens) / (max_actions * 3))
            else:
                confidence = 0.0
        
        # Decode actions
        actions = self.decode_action_sequence(generated_tokens)
        
        return actions, confidence
    
    def evaluate_single_problem(self, problem_data: Dict) -> Dict:
        """Evaluate model on a single ARC problem"""
        problem_id = problem_data['id']
        
        # For Re-ARC, use all examples
        test_cases = problem_data.get('train', [])
        
        # Limit test cases if specified
        max_test_samples = self.config.get('max_test_samples')
        if max_test_samples and len(test_cases) > max_test_samples:
            test_cases = test_cases[:max_test_samples]
        
        results = []
        
        for test_idx, test_case in enumerate(tqdm(test_cases, desc=f"Problem {problem_id}", leave=False)):
            test_input = test_case['input']
            test_output = test_case['output']
            
            try:
                # Generate action sequence
                predicted_actions, confidence = self.generate_action_sequence(
                    test_input, 
                    max_actions=self.config.get('problem_max_actions', {}).get(str(problem_id), 20)
                )
                
                # Execute actions
                predicted_grid = self.executor.execute_action_sequence(
                    test_input, predicted_actions
                )
                
                # Check correctness
                is_correct = np.array_equal(np.array(predicted_grid), np.array(test_output))
                
                result = {
                    'test_idx': test_idx,
                    'predicted_actions': predicted_actions,
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'num_actions': len(predicted_actions)
                }
                
                results.append(result)
                
                if test_idx % 10 == 0:
                    self.logger.info(
                        f"Problem {problem_id}, Test {test_idx}: "
                        f"{'CORRECT' if is_correct else 'INCORRECT'} "
                        f"(actions: {len(predicted_actions)})"
                    )
                
            except Exception as e:
                self.logger.error(f"Error evaluating problem {problem_id}, test {test_idx}: {e}")
                result = {
                    'test_idx': test_idx,
                    'error': str(e),
                    'is_correct': False,
                    'confidence': 0.0
                }
                results.append(result)
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        accuracy = correct_count / len(results) if results else 0.0
        
        problem_result = {
            'problem_id': problem_id,
            'test_results': results,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(results)
        }
        
        return problem_result
    
    def load_rearc_problems(self, problem_ids: List[int]) -> List[Dict]:
        """Load Re-ARC problems"""
        id_to_hex = {
            86: "25ff71a9",
            128: "5582e5ca", 
            139: "6150a2bd",
            149: "67a3c6ac",
            154: "68b16354",
            178: "74dd1130",
            240: "9dfd6313",
            379: "ed36ccf7"
        }
        
        problems = []
        rearc_dir = self.config['rearc_data_dir']
        
        for problem_id in problem_ids:
            if problem_id not in id_to_hex:
                self.logger.warning(f"No hex mapping for problem ID {problem_id}")
                continue
                
            hex_filename = id_to_hex[problem_id]
            problem_file = os.path.join(rearc_dir, f"{hex_filename}.json")
            
            if os.path.exists(problem_file):
                with open(problem_file, 'r') as f:
                    raw_data = json.load(f)
                    problem_data = {
                        'train': raw_data,
                        'id': problem_id,
                        'hex_id': hex_filename
                    }
                    problems.append(problem_data)
                    self.logger.info(f"Loaded problem {problem_id} with {len(raw_data)} examples")
            else:
                self.logger.warning(f"Problem file not found: {problem_file}")
        
        return problems
    
    def run_evaluation(self, problem_ids: List[int] = None) -> Dict:
        """Run full evaluation on Re-ARC problems"""
        if problem_ids is None:
            problem_ids = self.config.get('eval_problems', [86, 139, 178, 149, 154, 240, 379])
        
        self.logger.info(f"Starting evaluation on problems: {problem_ids}")
        
        # Load problems
        problems = self.load_rearc_problems(problem_ids)
        if not problems:
            raise ValueError("No problems loaded for evaluation")
        
        # Evaluate each problem
        all_results = []
        total_correct = 0
        total_tests = 0
        
        for problem in tqdm(problems, desc="Evaluating problems"):
            problem_result = self.evaluate_single_problem(problem)
            all_results.append(problem_result)
            
            total_correct += problem_result['correct_count']
            total_tests += problem_result['total_count']
        
        # Calculate overall statistics
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        
        evaluation_summary = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_tests': total_tests,
            'problem_results': all_results,
            'model_config': self.config,
            'evaluation_problems': problem_ids
        }
        
        # Save results
        results_file = os.path.join(self.config['results_dir'], 'evaluation_30x30_results.json')
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        self.logger.info(f"\n=== Evaluation Summary ===")
        self.logger.info(f"Overall accuracy: {overall_accuracy:.3f} ({total_correct}/{total_tests})")
        self.logger.info(f"Results saved to: {results_file}")
        
        # Print per-problem summary
        self.logger.info("\nPer-problem results:")
        for result in all_results:
            self.logger.info(
                f"  Problem {result['problem_id']}: "
                f"{result['accuracy']:.3f} ({result['correct_count']}/{result['total_count']})"
            )
        
        return evaluation_summary

def main():
    parser = argparse.ArgumentParser(description="30x30 ARC Trajectory Transformer Inference")
    parser.add_argument("--config", type=str, default="./processed_data_30x30/config_30x30.yaml",
                       help="Config file path")
    parser.add_argument("--model_path", type=str, default="./models_30x30/checkpoint_30x30_best.pt",
                       help="Model checkpoint path")
    parser.add_argument("--problems", type=int, nargs="+", default=None,
                       help="Problem IDs to evaluate")
    parser.add_argument("--gpu", type=int, default=5,
                       help="GPU device to use")
    
    args = parser.parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize inference engine
    inference_engine = ARC30x30Inference(config, args.model_path)
    
    # Run evaluation
    results = inference_engine.run_evaluation(args.problems)
    
    print(f"\n=== Final Results ===")
    print(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
    print(f"Correct: {results['total_correct']}/{results['total_tests']}")

if __name__ == "__main__":
    main()