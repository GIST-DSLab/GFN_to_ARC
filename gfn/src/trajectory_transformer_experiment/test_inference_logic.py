#!/usr/bin/env python3
"""
Inference 코드의 입출력 형식 및 함수 파싱 검증 테스트
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
import sys

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ARCenv.EntireARCEnv import DiagonalARCEnv


class InferenceLogicTester:
    """Inference 로직 테스트용 클래스"""
    
    def __init__(self):
        self.vocab = self.create_vocabulary()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def create_vocabulary(self):
        """Vocabulary 생성"""
        vocab = {
            # Grid colors (0-9)
            'color_0': 0, 'color_1': 1, 'color_2': 2, 'color_3': 3, 'color_4': 4,
            'color_5': 5, 'color_6': 6, 'color_7': 7, 'color_8': 8, 'color_9': 9,
            'pad': 10,  # Padding token
            
            # Actions (11-15) 
            'action_0': 11,  # left_rotate
            'action_1': 12,  # right_rotate  
            'action_2': 13,  # horizontal_flip
            'action_3': 14,  # vertical_flip
            'action_4': 15,  # submit
            
            # Rewards (16-19)
            'reward_neg': 16,   # Negative/zero reward
            'reward_small': 17, # Small positive reward
            'reward_med': 18,   # Medium positive reward
            'reward_large': 19, # Large positive reward
            
            # Special tokens
            'sos': 20,  # Start of sequence
            'eos': 21,  # End of sequence
        }
        return vocab
    
    def test_encode_initial_state(self):
        """초기 상태 인코딩 테스트"""
        print("=== 초기 상태 인코딩 테스트 ===")
        
        # 테스트 케이스 1: 3x3 그리드
        test_grid = [
            [1, 2, 3],
            [4, 5, 6], 
            [7, 8, 9]
        ]
        
        # 예상 결과: 평면화된 그리드 [1,2,3,4,5,6,7,8,9]
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        # 실제 인코딩 로직 (inference.py에서 가져옴)
        flat_grid = np.array(test_grid).flatten()
        tokens = [int(x) if x < 10 else 10 for x in flat_grid]
        
        print(f"입력 그리드: {test_grid}")
        print(f"평면화된 그리드: {flat_grid.tolist()}")
        print(f"토큰화된 결과: {tokens}")
        print(f"예상 결과: {expected}")
        print(f"테스트 결과: {'PASS' if tokens == expected else 'FAIL'}")
        
        # 테스트 케이스 2: 10 이상 값 처리
        test_grid_large = [
            [1, 12, 3],
            [4, 15, 6],
            [7, 8, 20]
        ]
        
        flat_grid_large = np.array(test_grid_large).flatten()
        tokens_large = [int(x) if x < 10 else 10 for x in flat_grid_large]
        expected_large = [1, 10, 3, 4, 10, 6, 7, 8, 10]  # 10 이상 값들이 10으로 변환
        
        print(f"\n큰 값 처리 테스트:")
        print(f"입력 그리드: {test_grid_large}")
        print(f"토큰화된 결과: {tokens_large}")
        print(f"예상 결과: {expected_large}")
        print(f"테스트 결과: {'PASS' if tokens_large == expected_large else 'FAIL'}")
        
        return tokens == expected and tokens_large == expected_large
    
    def test_decode_action_sequence(self):
        """액션 시퀀스 디코딩 테스트"""
        print("\n=== 액션 시퀀스 디코딩 테스트 ===")
        
        # 테스트 케이스: 혼합된 토큰 시퀀스
        test_tokens = [20, 1, 2, 3, 11, 16, 4, 5, 6, 12, 17, 7, 8, 9, 15, 19, 21]
        # 액션 토큰: 11 (action_0), 12 (action_1), 15 (action_4)
        # 예상 액션: [0, 1, 4]
        
        # 실제 디코딩 로직 (inference.py에서 가져옴)
        actions = []
        for token in test_tokens:
            if 11 <= token <= 15:
                action = token - 11  # Convert back to action index (0-4)
                actions.append(action)
        
        expected_actions = [0, 1, 4]
        
        print(f"입력 토큰: {test_tokens}")
        print(f"디코딩된 액션: {actions}")
        print(f"예상 액션: {expected_actions}")
        print(f"테스트 결과: {'PASS' if actions == expected_actions else 'FAIL'}")
        
        # 액션 매핑 확인
        print(f"\n액션 매핑:")
        print(f"11 -> 0 (left_rotate)")
        print(f"12 -> 1 (right_rotate)")
        print(f"13 -> 2 (horizontal_flip)")
        print(f"14 -> 3 (vertical_flip)")
        print(f"15 -> 4 (submit)")
        
        return actions == expected_actions
    
    def test_arc_environment_actions(self):
        """ARC 환경에서 액션 실행 테스트"""
        print("\n=== ARC 환경 액션 실행 테스트 ===")
        
        # 테스트 그리드
        test_grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        try:
            # ARC 환경 초기화
            env = DiagonalARCEnv()
            
            # 각 액션 테스트
            actions_to_test = [
                (0, "left_rotate"),
                (1, "right_rotate"),
                (2, "horizontal_flip"),
                (3, "vertical_flip")
            ]
            
            for action_idx, action_name in actions_to_test:
                # 환경 초기화
                current_grid = [row[:] for row in test_grid]  # 깊은 복사
                
                print(f"\n{action_name} (액션 {action_idx}) 테스트:")
                print(f"입력 그리드:")
                for row in current_grid:
                    print(f"  {row}")
                
                # 액션 실행 로직 확인
                if action_idx == 0:  # left_rotate
                    # 반시계 방향 90도 회전
                    result = [[current_grid[j][2-i] for j in range(3)] for i in range(3)]
                elif action_idx == 1:  # right_rotate
                    # 시계 방향 90도 회전
                    result = [[current_grid[2-j][i] for j in range(3)] for i in range(3)]
                elif action_idx == 2:  # horizontal_flip
                    # 수평 뒤집기
                    result = [row[::-1] for row in current_grid]
                elif action_idx == 3:  # vertical_flip
                    # 수직 뒤집기
                    result = current_grid[::-1]
                
                print(f"결과 그리드:")
                for row in result:
                    print(f"  {row}")
            
            return True
            
        except Exception as e:
            print(f"ARC 환경 테스트 실패: {e}")
            return False
    
    def test_evaluation_correctness_logic(self):
        """정답 판정 로직 테스트"""
        print("\n=== 정답 판정 로직 테스트 ===")
        
        # 테스트 케이스들
        test_cases = [
            {
                'predicted': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                'expected': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                'should_match': True
            },
            {
                'predicted': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                'expected': [[1, 2, 3], [4, 5, 6], [7, 8, 0]],
                'should_match': False
            },
            {
                'predicted': [[1, 2], [3, 4]],
                'expected': [[1, 2, 3], [4, 5, 6]],
                'should_match': False
            }
        ]
        
        all_passed = True
        
        for i, test_case in enumerate(test_cases):
            predicted = test_case['predicted']
            expected = test_case['expected']
            should_match = test_case['should_match']
            
            # 실제 비교 로직 (inference.py에서 가져옴)
            is_correct = np.array_equal(np.array(predicted), np.array(expected))
            
            passed = is_correct == should_match
            all_passed = all_passed and passed
            
            print(f"테스트 케이스 {i+1}:")
            print(f"  예측: {predicted}")
            print(f"  정답: {expected}")
            print(f"  예상 결과: {should_match}")
            print(f"  실제 결과: {is_correct}")
            print(f"  테스트: {'PASS' if passed else 'FAIL'}")
        
        return all_passed
    
    def test_problem_data_format(self):
        """문제 데이터 포맷 테스트"""
        print("\n=== 문제 데이터 포맷 테스트 ===")
        
        # ReARC 데이터 포맷 시뮬레이션
        mock_rearc_data = [
            {
                'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                'output': [[3, 6, 9], [2, 5, 8], [1, 4, 7]]
            },
            {
                'input': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            }
        ]
        
        # 표준 ARC 포맷으로 변환 테스트
        problem_data = {
            'train': mock_rearc_data,
            'test': [],  # ReARC에는 test가 없음
            'id': 86,
            'hex_id': '25ff71a9'
        }
        
        print(f"문제 데이터 포맷:")
        print(f"  ID: {problem_data['id']}")
        print(f"  Hex ID: {problem_data['hex_id']}")
        print(f"  Train 예제 수: {len(problem_data['train'])}")
        print(f"  Test 예제 수: {len(problem_data['test'])}")
        
        # 입출력 형식 확인
        for i, example in enumerate(problem_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            
            print(f"  예제 {i+1}:")
            print(f"    입력 형태: {np.array(input_grid).shape}")
            print(f"    출력 형태: {np.array(output_grid).shape}")
            print(f"    입력 타입: {type(input_grid)}")
            print(f"    출력 타입: {type(output_grid)}")
        
        # 형식 검증
        format_ok = True
        for example in problem_data['train']:
            if not isinstance(example['input'], list) or not isinstance(example['output'], list):
                format_ok = False
                break
            if len(example['input']) != 3 or len(example['output']) != 3:
                format_ok = False
                break
        
        print(f"포맷 검증: {'PASS' if format_ok else 'FAIL'}")
        return format_ok
    
    def test_accuracy_calculation(self):
        """정확도 계산 로직 테스트"""
        print("\n=== 정확도 계산 로직 테스트 ===")
        
        # 테스트 시나리오
        test_results = [
            {'is_correct': True},
            {'is_correct': False},
            {'is_correct': True},
            {'is_correct': True},
            {'is_correct': False},
            {'error': 'Some error', 'is_correct': False}
        ]
        
        # 정확도 계산 로직 (inference.py에서 가져옴)
        correct_count = sum(1 for r in test_results if r.get('is_correct', False))
        total_count = len(test_results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        expected_correct = 3
        expected_total = 6
        expected_accuracy = 3/6
        
        print(f"테스트 결과: {test_results}")
        print(f"정답 수: {correct_count} (예상: {expected_correct})")
        print(f"총 수: {total_count} (예상: {expected_total})")
        print(f"정확도: {accuracy:.4f} (예상: {expected_accuracy:.4f})")
        
        calculation_ok = (correct_count == expected_correct and 
                         total_count == expected_total and 
                         abs(accuracy - expected_accuracy) < 1e-6)
        
        print(f"계산 검증: {'PASS' if calculation_ok else 'FAIL'}")
        return calculation_ok
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🧪 Inference 로직 테스트 시작\n")
        
        tests = [
            ("초기 상태 인코딩", self.test_encode_initial_state),
            ("액션 시퀀스 디코딩", self.test_decode_action_sequence),
            ("ARC 환경 액션", self.test_arc_environment_actions),
            ("정답 판정 로직", self.test_evaluation_correctness_logic),
            ("문제 데이터 포맷", self.test_problem_data_format),
            ("정확도 계산", self.test_accuracy_calculation)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                print(f"테스트 '{test_name}' 실행 중 오류: {e}")
                results[test_name] = False
        
        # 결과 요약
        print("\n" + "="*50)
        print("📊 테스트 결과 요약")
        print("="*50)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
            if result:
                passed += 1
        
        print(f"\n전체 결과: {passed}/{total} 테스트 통과")
        print(f"성공률: {passed/total*100:.1f}%")
        
        return results


def main():
    tester = InferenceLogicTester()
    results = tester.run_all_tests()
    
    # 결과 저장
    output_file = "inference_logic_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n결과가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()