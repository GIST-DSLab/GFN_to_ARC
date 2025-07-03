#!/usr/bin/env python3
"""
ARC Trajectory Transformer Inference Script
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

from models.arc_transformer import create_model
from utils.data_utils import create_vocabulary, flatten_grid_state, discretize_reward
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ARCenv.EntireARCEnv import DiagonalARCEnv


class ARCTrajectoryInference:
    """ARC Trajectory Transformer Inference Engine"""
    
    def __init__(self, config, model_path: str):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.vocab = create_vocabulary()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Setup logging
        self.setup_logging()
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Initialize ARC environment for action execution
        self.executor = self.create_executor()
        
        self.logger.info(f"Inference engine initialized on {self.device}")
    
    def create_executor(self):
        """Create action executor for ARC environment"""
        class ActionExecutor:
            def execute_action_sequence(self, initial_grid, actions):
                """Execute sequence of actions on grid"""
                current_grid = [row[:] for row in initial_grid]  # Deep copy
                
                for action in actions:
                    if action == 0:  # left_rotate
                        current_grid = [[current_grid[j][2-i] for j in range(3)] for i in range(3)]
                    elif action == 1:  # right_rotate
                        current_grid = [[current_grid[2-j][i] for j in range(3)] for i in range(3)]
                    elif action == 2:  # horizontal_flip
                        current_grid = [row[::-1] for row in current_grid]
                    elif action == 3:  # vertical_flip
                        current_grid = current_grid[::-1]
                    elif action == 4:  # submit
                        break  # Stop execution on submit
                
                return current_grid
        
        return ActionExecutor()
    
    def setup_logging(self):
        """Setup logging"""
        log_file = os.path.join(self.config['results_dir'], 'inference.log')
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
        model = create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model loaded successfully")
        self.logger.info(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"Best validation loss: {checkpoint.get('train_losses', [])[-1] if checkpoint.get('train_losses') else 'unknown'}")
        
        return model
    
    def encode_initial_state(self, grid_state: List[List[int]]) -> List[int]:
        """Encode initial grid state as tokens"""
        # Flatten 3x3 grid
        flat_grid = np.array(grid_state).flatten()
        
        # Convert to tokens (0-9 for colors, 10 for padding)
        tokens = [int(x) if x < 10 else 10 for x in flat_grid]
        
        return tokens
    
    def decode_action_sequence(self, tokens: List[int]) -> List[int]:
        """Extract action sequence from generated tokens"""
        actions = []
        
        # Find action tokens (11-15) in the sequence
        for token in tokens:
            if 11 <= token <= 15:
                action = token - 11  # Convert back to action index (0-4)
                actions.append(action)
        
        return actions
    
    def generate_action_sequence(self, initial_grid: List[List[int]], 
                                max_actions: int = 20) -> Tuple[List[int], float]:
        """
        Generate action sequence for given initial grid state
        
        Returns:
            actions: List of action indices
            confidence: Average prediction confidence
        """
        # Encode initial state
        initial_tokens = self.encode_initial_state(initial_grid)
        
        # Add start token
        input_ids = torch.tensor([self.vocab['sos']] + initial_tokens, 
                                device=self.device).unsqueeze(0)
        
        confidences = []
        
        with torch.no_grad():
            # Generate sequence
            generated = self.model.generate(
                input_ids,
                max_new_tokens=self.config.get('max_new_tokens', 32),
                temperature=self.config.get('temperature', 1.0),
                top_k=self.config.get('top_k', None),
                top_p=self.config.get('top_p', 0.9),
                pad_token_id=self.vocab['pad'],
                eos_token_id=self.vocab['eos']
            )
            
            # Extract generated tokens (excluding initial input)
            generated_tokens = generated[0, input_ids.size(1):].cpu().tolist()
            
            # Calculate confidence (simplified - using max probability)
            outputs = self.model(generated[0:1])
            logits = outputs['logits'][0, input_ids.size(1)-1:]
            probs = F.softmax(logits, dim=-1)
            token_confidences = torch.max(probs, dim=-1)[0].cpu().tolist()
            confidences.extend(token_confidences)
        
        # Decode actions from generated sequence
        actions = self.decode_action_sequence(generated_tokens)
        
        # Limit actions
        actions = actions[:max_actions]
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return actions, avg_confidence
    
    def evaluate_single_problem(self, problem_data: Dict) -> Dict:
        """Evaluate model on a single ARC problem"""
        problem_id = problem_data['id']
        # For ReARC, use train data as test cases since there's no separate test data
        test_cases = problem_data.get('test', [])
        if not test_cases:
            test_cases = problem_data.get('train', [])[:3]  # Use first 3 train examples as test
        
        results = []
        
        for test_idx, test_case in enumerate(test_cases):
            test_input = test_case['input']
            test_output = test_case['output']
            
            try:
                # Generate action sequence
                predicted_actions, confidence = self.generate_action_sequence(
                    test_input, 
                    max_actions=self.config.get('max_actions', 20)
                )
                
                # Execute actions on the grid
                predicted_grid = self.executor.execute_action_sequence(
                    test_input, predicted_actions
                )
                
                # Check correctness
                is_correct = np.array_equal(np.array(predicted_grid), np.array(test_output))
                
                result = {
                    'test_idx': test_idx,
                    'predicted_actions': predicted_actions,
                    'predicted_grid': predicted_grid,
                    'expected_grid': test_output,
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'num_actions': len(predicted_actions)
                }
                
                results.append(result)
                
                self.logger.info(
                    f"Problem {problem_id}, Test {test_idx}: "
                    f"{'CORRECT' if is_correct else 'INCORRECT'} "
                    f"(confidence: {confidence:.3f}, actions: {len(predicted_actions)})"
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
        
        # Calculate problem-level accuracy
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
        """Load ReARC problems"""
        # Mapping from problem ID to hex filename
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
                self.logger.warning(f"No hex mapping found for problem ID {problem_id}")
                continue
                
            hex_filename = id_to_hex[problem_id]
            problem_file = os.path.join(rearc_dir, f"{hex_filename}.json")
            
            if os.path.exists(problem_file):
                with open(problem_file, 'r') as f:
                    raw_data = json.load(f)
                    # Re-ARC format: list of examples with input/output
                    # Convert to standard ARC format
                    problem_data = {
                        'train': raw_data,  # All examples as training data
                        'test': [],  # No test data in re-arc format
                        'id': problem_id,
                        'hex_id': hex_filename
                    }
                    problems.append(problem_data)
                    self.logger.info(f"Loaded problem {problem_id} ({hex_filename}) with {len(raw_data)} examples")
            else:
                self.logger.warning(f"Problem file not found: {problem_file}")
        
        return problems
    
    def run_evaluation(self, problem_ids: List[int] = None) -> Dict:
        """Run full evaluation on ReARC problems"""
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
        
        for problem in tqdm(problems, desc="Evaluating problems", unit="problem"):
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
        results_file = os.path.join(self.config['results_dir'], 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        self.logger.info(f"Evaluation completed!")
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
    parser = argparse.ArgumentParser(description="ARC Trajectory Transformer Inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Configuration file path")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model checkpoint (overrides config)")
    parser.add_argument("--problems", type=int, nargs="+", default=None,
                       help="Problem IDs to evaluate (default: all)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for inference (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.device:
        config['device'] = args.device
    if args.output_dir:
        config['results_dir'] = args.output_dir
    if args.model_path:
        config['model_load_path'] = args.model_path
    else:
        # Use default model path from config
        config['model_load_path'] = os.path.join(config['model_save_dir'], "arc_transformer_best.pt")
    
    # Initialize inference engine
    inference_engine = ARCTrajectoryInference(config, config['model_load_path'])
    
    try:
        # Run evaluation
        results = inference_engine.run_evaluation(args.problems)
        
        print(f"\n=== Evaluation Summary ===")
        print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
        print(f"Correct: {results['total_correct']}/{results['total_tests']}")
        print(f"Results saved to: {os.path.join(args.output_dir, 'evaluation_results.json')}")
        
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()