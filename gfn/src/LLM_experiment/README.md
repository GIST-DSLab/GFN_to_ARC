# LLM Experiment for ARC Action Sequence Learning

이 프로젝트는 GFlowNet trajectory 데이터를 사용하여 LLM이 ARC 문제의 action sequence를 학습하고 예측할 수 있도록 하는 실험입니다.

## 📁 프로젝트 구조

```
LLM_experiment/
├── configs/
│   └── config.yaml              # 실험 설정 파일
├── data_preprocessing.py        # 데이터 전처리 (padding 제거, action 매핑)
├── training.py                  # LLM 학습 (trajectory → action sequence)
├── inference.py                 # 추론 및 평가 (ReARC 데이터셋)
├── run_experiment.py           # 전체 실험 실행 스크립트
├── utils.py                    # 공통 유틸리티 함수
├── requirements.txt            # 필요한 패키지 목록
└── README.md                   # 이 파일
```

## 🚀 실행 방법

### 1. 전체 실험 실행 (권장)

```bash
cd /home/ubuntu/GFN_to_ARC/gfn/src/LLM_experiment
python run_experiment.py
```

### 2. 단계별 실행

```bash
# 데이터 전처리만
python run_experiment.py --preprocessing-only

# 학습만 (전처리 완료 후)
python run_experiment.py --training-only

# 추론만 (학습 완료 후)
python run_experiment.py --inference-only
```

### 3. 개별 모듈 실행

```bash
# 데이터 전처리
python data_preprocessing.py

# 모델 학습
python training.py

# 추론 및 평가
python inference.py
```

## ⚙️ 설정

`configs/config.yaml` 파일에서 다음 설정들을 조정할 수 있습니다:

- **모델 설정**: LLM 모델 선택 (기본: DialoGPT-small)
- **학습 파라미터**: batch size, learning rate, epochs 등
- **액션 매핑**: GFlowNet action → ARC action 매핑
- **데이터 경로**: trajectory 데이터 및 ReARC 데이터 경로

## 📊 데이터 흐름

1. **입력 데이터**: GFlowNet trajectory JSON 파일들
   - 위치: `../trajectories_output/problem_*/trajectories_0_1000.json`
   - 문제들: 52, 86, 128, 139, 149, 154, 178, 240, 379

2. **전처리 과정**:
   - 30x30 padding 제거하여 실제 그리드 크기로 변환
   - Action ID 매핑 (0: 왼쪽 회전, 1: 오른쪽 회전, 2: 수평 뒤집기, 3: 수직 뒤집기, 4: 제출)
   - LLM 학습용 프롬프트 형태로 변환

3. **학습 데이터 형태**:
   ```
   Input: [[2,2,1],[1,5,1],[5,2,2]]
   Output: [[2,1,5],[2,5,2],[1,1,2]]
   Actions: [left_rotate,submit]
   ```

4. **평가 데이터**: ReARC 데이터셋
   - 위치: `/home/ubuntu/gflownet-llm/data/re-arc/arc_original/training/`
   - 학습에 사용된 문제들의 ARC 원본 데이터

## 🔧 Action 매핑

| GFlowNet Action | ARC Action | 설명 |
|-----------------|------------|------|
| 0 | 25 | 왼쪽 회전 (90도 반시계) |
| 1 | 24 | 오른쪽 회전 (90도 시계) |
| 2 | 26 | 수평 뒤집기 |
| 3 | 27 | 수직 뒤집기 |
| 4 | 34 | 제출 |

## 📈 평가 방법

1. **LLM 예측**: 입력-출력 그리드 쌍에 대해 action sequence 생성
2. **Action 실행**: 예측된 action들을 실제로 입력 그리드에 적용
3. **정확도 계산**: 최종 결과가 목표 출력과 일치하는지 확인

## 📋 결과 파일

실험 완료 후 다음 파일들이 생성됩니다:

- `./results/evaluation_results.json`: 상세한 평가 결과
- `./results/experiment_summary.json`: 실험 요약
- `./models/`: 학습된 모델 파일들
- `./processed_data/`: 전처리된 데이터
- `./logs/`: 실행 로그

## 🔍 문제 해결

### 일반적인 오류들

1. **CUDA 메모리 부족**: `config.yaml`에서 `batch_size` 줄이기
2. **데이터 파일 없음**: trajectory 데이터가 생성되었는지 확인
3. **패키지 설치**: `pip install -r requirements.txt`

### 로그 확인

```bash
# 전체 실험 로그
tail -f logs/experiment.log

# 전처리 로그
tail -f processed_data/preprocessing.log

# 학습 로그
tail -f models/training.log
```

## 🎯 실험 목표

이 실험의 목표는 다음과 같습니다:

1. GFlowNet이 생성한 trajectory 데이터로 LLM 학습
2. 학습된 LLM이 새로운 ARC 문제에서 올바른 action sequence 예측
3. 예측된 action을 실제 실행하여 정확한 결과 생성 확인
4. ReARC 데이터셋에서의 성능 평가

## 📚 참고사항

- 학습 시간: 모델 크기와 데이터양에 따라 수 시간 소요 가능
- GPU 권장: CUDA 사용 가능한 환경에서 실행 권장
- 메모리 요구사항: 최소 8GB RAM, 4GB VRAM 권장