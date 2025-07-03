#!/usr/bin/env python3
"""
Inference 코드에서 executor 없이 정답/오답 처리 로직 테스트
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
import sys


class MockExecutor:
    """ARC 환경 액션 실행을 시뮬레이션하는 Mock 클래스"""
    
    def execute_action_sequence(self, initial_grid: List[List[int]], actions: List[int]) -> List[List[int]]:
        """액션 시퀀스를 실행하여 최종 그리드 반환"""
        current_grid = [row[:] for row in initial_grid]  # 깊은 복사
        
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
                break  # 제출 액션에서 중단
        
        return current_grid


class InferenceProcessTester:
    """Inference 프로세스 테스트"""
    
    def __init__(self):
        self.executor = MockExecutor()
        self.vocab = self.create_vocabulary()
        
    def create_vocabulary(self):
        """Vocabulary 생성"""
        vocab = {
            'color_0': 0, 'color_1': 1, 'color_2': 2, 'color_3': 3, 'color_4': 4,
            'color_5': 5, 'color_6': 6, 'color_7': 7, 'color_8': 8, 'color_9': 9,
            'pad': 10, 'action_0': 11, 'action_1': 12, 'action_2': 13, 'action_3': 14, 'action_4': 15,
            'reward_neg': 16, 'reward_small': 17, 'reward_med': 18, 'reward_large': 19,
            'sos': 20, 'eos': 21,
        }
        return vocab
    
    def test_problem_evaluation_pipeline(self):
        """문제 평가 파이프라인 테스트"""
        print("=== 문제 평가 파이프라인 테스트 ===")
        
        # 테스트 문제 생성
        test_problem = {
            'id': 86,
            'test': [
                {
                    'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    'output': [[3, 6, 9], [2, 5, 8], [1, 4, 7]]  # left_rotate 결과
                },
                {
                    'input': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                    'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]  # 변화 없음
                }
            ]
        }
        
        # 예측 액션 시퀀스 (모의)
        predicted_actions_list = [
            [0, 4],  # left_rotate + submit -> 정답
            [2, 4],  # horizontal_flip + submit -> 오답
        ]
        
        results = []
        
        for test_idx, test_case in enumerate(test_problem['test']):
            test_input = test_case['input']
            test_output = test_case['output']
            predicted_actions = predicted_actions_list[test_idx]
            
            print(f"\n테스트 케이스 {test_idx + 1}:")
            print(f"입력 그리드: {test_input}")
            print(f"예상 출력: {test_output}")
            print(f"예측 액션: {predicted_actions}")
            
            # 액션 실행
            predicted_grid = self.executor.execute_action_sequence(test_input, predicted_actions)
            print(f"예측 출력: {predicted_grid}")
            
            # 정답 확인
            is_correct = np.array_equal(np.array(predicted_grid), np.array(test_output))
            print(f"정답 여부: {is_correct}")
            
            result = {
                'test_idx': test_idx,
                'predicted_actions': predicted_actions,
                'predicted_grid': predicted_grid,
                'expected_grid': test_output,
                'is_correct': is_correct,
                'confidence': 0.8,  # 모의 신뢰도
                'num_actions': len(predicted_actions)
            }
            
            results.append(result)
        
        # 문제 수준 정확도 계산
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        accuracy = correct_count / len(results) if results else 0.0
        
        problem_result = {
            'problem_id': test_problem['id'],
            'test_results': results,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(results)
        }
        
        print(f"\n문제 {test_problem['id']} 결과:")
        print(f"정답 수: {correct_count}/{len(results)}")
        print(f"정확도: {accuracy:.3f}")
        
        return problem_result
    
    def test_error_handling(self):
        """오류 처리 테스트"""
        print("\n=== 오류 처리 테스트 ===")
        
        # 테스트 케이스: 오류 상황들
        error_cases = [
            {
                'name': '잘못된 액션 인덱스',
                'actions': [0, 5, 4],  # 5는 유효하지 않은 액션
                'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            },
            {
                'name': '빈 액션 리스트',
                'actions': [],
                'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            },
            {
                'name': '잘못된 그리드 형식',
                'actions': [0, 4],
                'input': [[1, 2], [3, 4]]  # 2x2 그리드
            }
        ]
        
        for case in error_cases:
            print(f"\n테스트: {case['name']}")
            print(f"입력: {case['input']}")
            print(f"액션: {case['actions']}")
            
            try:
                result = self.executor.execute_action_sequence(case['input'], case['actions'])
                print(f"결과: {result}")
                print("상태: 성공")
            except Exception as e:
                print(f"오류: {e}")
                print("상태: 오류 발생")
    
    def test_action_sequence_validation(self):
        """액션 시퀀스 검증 테스트"""
        print("\n=== 액션 시퀀스 검증 테스트 ===")
        
        # 테스트 케이스들
        test_cases = [
            {
                'name': '유효한 액션 시퀀스',
                'tokens': [20, 1, 2, 3, 11, 16, 4, 5, 6, 12, 17, 7, 8, 9, 15, 19, 21],
                'expected_actions': [0, 1, 4]
            },
            {
                'name': '액션 없는 시퀀스',
                'tokens': [20, 1, 2, 3, 16, 4, 5, 6, 17, 7, 8, 9, 19, 21],
                'expected_actions': []
            },
            {
                'name': '연속된 액션',
                'tokens': [20, 11, 12, 13, 14, 15, 21],
                'expected_actions': [0, 1, 2, 3, 4]
            }
        ]
        
        for case in test_cases:
            print(f"\n테스트: {case['name']}")
            print(f"토큰: {case['tokens']}")
            
            # 액션 디코딩
            actions = []
            for token in case['tokens']:
                if 11 <= token <= 15:
                    action = token - 11
                    actions.append(action)
            
            print(f"디코딩된 액션: {actions}")
            print(f"예상 액션: {case['expected_actions']}")
            print(f"일치 여부: {actions == case['expected_actions']}")
    
    def test_rearc_data_loading_simulation(self):
        """ReARC 데이터 로딩 시뮬레이션"""
        print("\n=== ReARC 데이터 로딩 시뮬레이션 ===")
        
        # ID to hex 매핑 (inference.py에서)
        id_to_hex = {
            86: "25ff71a9",
            139: "6150a2bd",
            178: "74dd1130",
            149: "67a3c6ac",
            154: "68b16354",
            240: "9dfd6313",
            379: "ed36ccf7"
        }
        
        # 문제 ID 리스트
        problem_ids = [86, 139, 999]  # 999는 존재하지 않는 ID
        
        loaded_problems = []
        
        for problem_id in problem_ids:
            print(f"\n문제 ID {problem_id} 로딩:")
            
            if problem_id not in id_to_hex:
                print(f"  경고: 문제 ID {problem_id}에 대한 hex 매핑이 없습니다")
                continue
            
            hex_filename = id_to_hex[problem_id]
            print(f"  Hex 파일명: {hex_filename}")
            
            # 가상의 raw 데이터
            mock_raw_data = [
                {
                    'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    'output': [[3, 6, 9], [2, 5, 8], [1, 4, 7]]
                },
                {
                    'input': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                    'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
                }
            ]
            
            # 표준 ARC 포맷으로 변환
            problem_data = {
                'train': mock_raw_data,
                'test': [],
                'id': problem_id,
                'hex_id': hex_filename
            }
            
            loaded_problems.append(problem_data)
            print(f"  로딩 완료: {len(mock_raw_data)}개 예제")
        
        print(f"\n총 {len(loaded_problems)}개 문제 로딩됨")
        return loaded_problems
    
    def test_full_evaluation_pipeline(self):
        """전체 평가 파이프라인 테스트"""
        print("\n=== 전체 평가 파이프라인 테스트 ===")
        
        # 문제 데이터 로딩
        problems = self.test_rearc_data_loading_simulation()
        
        # 모든 문제에 대해 평가
        all_results = []
        total_correct = 0
        total_tests = 0
        
        for problem in problems:
            print(f"\n문제 {problem['id']} 평가:")
            
            # 각 예제를 테스트 케이스로 사용 (ReARC는 test 케이스가 없으므로)
            test_cases = problem['train']
            results = []
            
            for test_idx, test_case in enumerate(test_cases):
                # 가상의 예측 액션 (실제로는 모델이 생성)
                predicted_actions = [0, 4] if test_idx == 0 else [2, 4]
                
                # 액션 실행
                predicted_grid = self.executor.execute_action_sequence(
                    test_case['input'], predicted_actions
                )
                
                # 정답 확인
                is_correct = np.array_equal(np.array(predicted_grid), np.array(test_case['output']))
                
                result = {
                    'test_idx': test_idx,
                    'predicted_actions': predicted_actions,
                    'is_correct': is_correct,
                    'confidence': 0.8
                }
                
                results.append(result)
                print(f"  테스트 {test_idx}: {'정답' if is_correct else '오답'}")
            
            # 문제 수준 정확도
            correct_count = sum(1 for r in results if r['is_correct'])
            accuracy = correct_count / len(results) if results else 0.0
            
            problem_result = {
                'problem_id': problem['id'],
                'test_results': results,
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': len(results)
            }
            
            all_results.append(problem_result)
            total_correct += correct_count
            total_tests += len(results)
            
            print(f"  정확도: {accuracy:.3f} ({correct_count}/{len(results)})")
        
        # 전체 통계
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        
        print(f"\n전체 평가 결과:")
        print(f"전체 정확도: {overall_accuracy:.3f} ({total_correct}/{total_tests})")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_tests': total_tests,
            'problem_results': all_results
        }
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🧪 Inference 정답/오답 처리 로직 테스트 시작\n")
        
        tests = [
            ("문제 평가 파이프라인", self.test_problem_evaluation_pipeline),
            ("오류 처리", self.test_error_handling),
            ("액션 시퀀스 검증", self.test_action_sequence_validation),
            ("ReARC 데이터 로딩", self.test_rearc_data_loading_simulation),
            ("전체 평가 파이프라인", self.test_full_evaluation_pipeline)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"테스트: {test_name}")
            print('='*50)
            
            try:
                result = test_func()
                print(f"✅ {test_name} 완료")
            except Exception as e:
                print(f"❌ {test_name} 실패: {e}")
                import traceback
                traceback.print_exc()


def main():
    tester = InferenceProcessTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()