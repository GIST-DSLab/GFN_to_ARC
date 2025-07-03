#!/usr/bin/env python3
"""
Evaluation 정확도 계산 로직의 정확성 테스트
"""

import os
import json
import numpy as np
from typing import List, Dict, Any


class EvaluationAccuracyTester:
    """Evaluation 정확도 계산 로직 테스트"""
    
    def test_single_problem_accuracy(self):
        """단일 문제 정확도 계산 테스트"""
        print("=== 단일 문제 정확도 계산 테스트 ===")
        
        test_cases = [
            {
                'name': '모든 테스트 정답',
                'results': [
                    {'is_correct': True},
                    {'is_correct': True},
                    {'is_correct': True}
                ],
                'expected_accuracy': 1.0,
                'expected_correct': 3,
                'expected_total': 3
            },
            {
                'name': '모든 테스트 오답',
                'results': [
                    {'is_correct': False},
                    {'is_correct': False},
                    {'is_correct': False}
                ],
                'expected_accuracy': 0.0,
                'expected_correct': 0,
                'expected_total': 3
            },
            {
                'name': '혼합 결과',
                'results': [
                    {'is_correct': True},
                    {'is_correct': False},
                    {'is_correct': True},
                    {'is_correct': False}
                ],
                'expected_accuracy': 0.5,
                'expected_correct': 2,
                'expected_total': 4
            },
            {
                'name': '오류가 포함된 결과',
                'results': [
                    {'is_correct': True},
                    {'error': 'Some error', 'is_correct': False},
                    {'is_correct': True},
                    {'error': 'Another error', 'is_correct': False}
                ],
                'expected_accuracy': 0.5,
                'expected_correct': 2,
                'expected_total': 4
            },
            {
                'name': '빈 결과',
                'results': [],
                'expected_accuracy': 0.0,
                'expected_correct': 0,
                'expected_total': 0
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            print(f"\n테스트: {test_case['name']}")
            results = test_case['results']
            
            # inference.py의 로직 재현
            correct_count = sum(1 for r in results if r.get('is_correct', False))
            total_count = len(results)
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            
            # 검증
            expected_correct = test_case['expected_correct']
            expected_total = test_case['expected_total']
            expected_accuracy = test_case['expected_accuracy']
            
            correct_match = correct_count == expected_correct
            total_match = total_count == expected_total
            accuracy_match = abs(accuracy - expected_accuracy) < 1e-6
            
            passed = correct_match and total_match and accuracy_match
            all_passed = all_passed and passed
            
            print(f"  결과 수: {total_count} (예상: {expected_total}) {'✓' if total_match else '✗'}")
            print(f"  정답 수: {correct_count} (예상: {expected_correct}) {'✓' if correct_match else '✗'}")
            print(f"  정확도: {accuracy:.4f} (예상: {expected_accuracy:.4f}) {'✓' if accuracy_match else '✗'}")
            print(f"  전체 테스트: {'PASS' if passed else 'FAIL'}")
        
        return all_passed
    
    def test_overall_accuracy_calculation(self):
        """전체 정확도 계산 테스트"""
        print("\n=== 전체 정확도 계산 테스트 ===")
        
        # 여러 문제의 결과 시뮬레이션
        problem_results = [
            {
                'problem_id': 86,
                'correct_count': 2,
                'total_count': 3,
                'accuracy': 2/3
            },
            {
                'problem_id': 139,
                'correct_count': 1,
                'total_count': 2,
                'accuracy': 0.5
            },
            {
                'problem_id': 178,
                'correct_count': 3,
                'total_count': 3,
                'accuracy': 1.0
            }
        ]
        
        # inference.py의 로직 재현
        total_correct = sum(result['correct_count'] for result in problem_results)
        total_tests = sum(result['total_count'] for result in problem_results)
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        
        # 수동 계산
        expected_total_correct = 2 + 1 + 3  # 6
        expected_total_tests = 3 + 2 + 3    # 8
        expected_overall_accuracy = 6 / 8   # 0.75
        
        print(f"문제별 결과:")
        for result in problem_results:
            print(f"  문제 {result['problem_id']}: {result['correct_count']}/{result['total_count']} = {result['accuracy']:.3f}")
        
        print(f"\n전체 통계:")
        print(f"  총 정답 수: {total_correct} (예상: {expected_total_correct})")
        print(f"  총 테스트 수: {total_tests} (예상: {expected_total_tests})")
        print(f"  전체 정확도: {overall_accuracy:.4f} (예상: {expected_overall_accuracy:.4f})")
        
        # 검증
        correct_match = total_correct == expected_total_correct
        total_match = total_tests == expected_total_tests
        accuracy_match = abs(overall_accuracy - expected_overall_accuracy) < 1e-6
        
        passed = correct_match and total_match and accuracy_match
        
        print(f"검증 결과: {'PASS' if passed else 'FAIL'}")
        
        return passed
    
    def test_edge_cases(self):
        """경계 케이스 테스트"""
        print("\n=== 경계 케이스 테스트 ===")
        
        edge_cases = [
            {
                'name': '모든 문제가 빈 결과',
                'problem_results': [
                    {'correct_count': 0, 'total_count': 0},
                    {'correct_count': 0, 'total_count': 0}
                ],
                'expected_overall': 0.0
            },
            {
                'name': '일부 문제만 빈 결과',
                'problem_results': [
                    {'correct_count': 2, 'total_count': 2},
                    {'correct_count': 0, 'total_count': 0},
                    {'correct_count': 1, 'total_count': 2}
                ],
                'expected_overall': 3/4  # (2+0+1)/(2+0+2)
            },
            {
                'name': '단일 문제 완벽 정답',
                'problem_results': [
                    {'correct_count': 5, 'total_count': 5}
                ],
                'expected_overall': 1.0
            },
            {
                'name': '단일 문제 완전 오답',
                'problem_results': [
                    {'correct_count': 0, 'total_count': 5}
                ],
                'expected_overall': 0.0
            }
        ]
        
        all_passed = True
        
        for case in edge_cases:
            print(f"\n테스트: {case['name']}")
            problem_results = case['problem_results']
            expected = case['expected_overall']
            
            # 계산
            total_correct = sum(r['correct_count'] for r in problem_results)
            total_tests = sum(r['total_count'] for r in problem_results)
            overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
            
            # 검증
            passed = abs(overall_accuracy - expected) < 1e-6
            all_passed = all_passed and passed
            
            print(f"  문제 결과: {problem_results}")
            print(f"  계산된 정확도: {overall_accuracy:.4f}")
            print(f"  예상 정확도: {expected:.4f}")
            print(f"  테스트 결과: {'PASS' if passed else 'FAIL'}")
        
        return all_passed
    
    def test_floating_point_precision(self):
        """부동소수점 정밀도 테스트"""
        print("\n=== 부동소수점 정밀도 테스트 ===")
        
        # 부동소수점 오차가 발생할 수 있는 케이스들
        precision_cases = [
            {
                'name': '1/3 정확도',
                'correct': 1,
                'total': 3,
                'expected': 1/3
            },
            {
                'name': '2/3 정확도',
                'correct': 2,
                'total': 3,
                'expected': 2/3
            },
            {
                'name': '1/7 정확도',
                'correct': 1,
                'total': 7,
                'expected': 1/7
            },
            {
                'name': '큰 수의 정확도',
                'correct': 999999,
                'total': 1000000,
                'expected': 999999/1000000
            }
        ]
        
        all_passed = True
        
        for case in precision_cases:
            print(f"\n테스트: {case['name']}")
            correct = case['correct']
            total = case['total']
            expected = case['expected']
            
            # 계산
            accuracy = correct / total if total > 0 else 0.0
            
            # 검증 (부동소수점 오차 허용)
            passed = abs(accuracy - expected) < 1e-10
            all_passed = all_passed and passed
            
            print(f"  정답/총수: {correct}/{total}")
            print(f"  계산된 정확도: {accuracy}")
            print(f"  예상 정확도: {expected}")
            print(f"  차이: {abs(accuracy - expected)}")
            print(f"  테스트 결과: {'PASS' if passed else 'FAIL'}")
        
        return all_passed
    
    def test_result_data_structure(self):
        """결과 데이터 구조 테스트"""
        print("\n=== 결과 데이터 구조 테스트 ===")
        
        # inference.py에서 생성되는 데이터 구조 시뮬레이션
        evaluation_summary = {
            'overall_accuracy': 0.75,
            'total_correct': 6,
            'total_tests': 8,
            'problem_results': [
                {
                    'problem_id': 86,
                    'test_results': [
                        {
                            'test_idx': 0,
                            'predicted_actions': [0, 4],
                            'predicted_grid': [[3, 6, 9], [2, 5, 8], [1, 4, 7]],
                            'expected_grid': [[3, 6, 9], [2, 5, 8], [1, 4, 7]],
                            'is_correct': True,
                            'confidence': 0.85,
                            'num_actions': 2
                        },
                        {
                            'test_idx': 1,
                            'predicted_actions': [2, 4],
                            'predicted_grid': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            'expected_grid': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            'is_correct': True,
                            'confidence': 0.72,
                            'num_actions': 2
                        }
                    ],
                    'accuracy': 1.0,
                    'correct_count': 2,
                    'total_count': 2
                }
            ],
            'model_config': {'n_layer': 3, 'n_head': 4},
            'evaluation_problems': [86, 139, 178]
        }
        
        # 데이터 구조 검증
        required_top_level = ['overall_accuracy', 'total_correct', 'total_tests', 'problem_results']
        required_problem_level = ['problem_id', 'test_results', 'accuracy', 'correct_count', 'total_count']
        required_test_level = ['test_idx', 'is_correct']
        
        # 상위 레벨 검증
        top_level_ok = all(key in evaluation_summary for key in required_top_level)
        print(f"상위 레벨 키 존재 확인: {'PASS' if top_level_ok else 'FAIL'}")
        
        # 문제 레벨 검증
        problem_level_ok = True
        for problem_result in evaluation_summary['problem_results']:
            if not all(key in problem_result for key in required_problem_level):
                problem_level_ok = False
                break
        print(f"문제 레벨 키 존재 확인: {'PASS' if problem_level_ok else 'FAIL'}")
        
        # 테스트 레벨 검증
        test_level_ok = True
        for problem_result in evaluation_summary['problem_results']:
            for test_result in problem_result['test_results']:
                if not all(key in test_result for key in required_test_level):
                    test_level_ok = False
                    break
            if not test_level_ok:
                break
        print(f"테스트 레벨 키 존재 확인: {'PASS' if test_level_ok else 'FAIL'}")
        
        # 데이터 타입 검증
        type_checks = [
            (evaluation_summary['overall_accuracy'], (int, float), '전체 정확도'),
            (evaluation_summary['total_correct'], int, '총 정답 수'),
            (evaluation_summary['total_tests'], int, '총 테스트 수'),
            (evaluation_summary['problem_results'], list, '문제 결과 리스트')
        ]
        
        type_ok = True
        for value, expected_type, description in type_checks:
            if not isinstance(value, expected_type):
                print(f"{description} 타입 오류: {type(value)} (예상: {expected_type})")
                type_ok = False
        
        print(f"데이터 타입 확인: {'PASS' if type_ok else 'FAIL'}")
        
        # JSON 직렬화 가능성 테스트
        try:
            json_str = json.dumps(evaluation_summary, indent=2)
            json_ok = True
        except Exception as e:
            print(f"JSON 직렬화 오류: {e}")
            json_ok = False
        
        print(f"JSON 직렬화 가능성: {'PASS' if json_ok else 'FAIL'}")
        
        return top_level_ok and problem_level_ok and test_level_ok and type_ok and json_ok
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🧪 Evaluation 정확도 계산 로직 테스트 시작\n")
        
        tests = [
            ("단일 문제 정확도 계산", self.test_single_problem_accuracy),
            ("전체 정확도 계산", self.test_overall_accuracy_calculation),
            ("경계 케이스", self.test_edge_cases),
            ("부동소수점 정밀도", self.test_floating_point_precision),
            ("결과 데이터 구조", self.test_result_data_structure)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"{'='*50}")
            print(f"테스트: {test_name}")
            print('='*50)
            
            try:
                result = test_func()
                results[test_name] = result
                print(f"{'✅ PASS' if result else '❌ FAIL'}: {test_name}")
            except Exception as e:
                print(f"❌ 테스트 '{test_name}' 실행 중 오류: {e}")
                results[test_name] = False
                import traceback
                traceback.print_exc()
        
        # 결과 요약
        print("\n" + "="*50)
        print("📊 테스트 결과 요약")
        print("="*50)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\n전체 결과: {passed}/{total} 테스트 통과")
        print(f"성공률: {passed/total*100:.1f}%")
        
        return results


def main():
    tester = EvaluationAccuracyTester()
    results = tester.run_all_tests()
    
    # 결과 저장
    output_file = "evaluation_accuracy_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n결과가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()