# LLM Experiment Final Model Results

이 폴더는 **GFN_to_ARC** 프로젝트의 LLM 실험 최종 결과와 재현 가능한 모델을 포함합니다.

## 📁 폴더 구조

```
final_model_result/
├── README.md                              # 이 파일
├── REPRODUCTION_TEST.md                   # 모델 재현 가능성 테스트 결과
├── MODEL_INFO.md                          # 모델 상세 정보
├── config_ddp_456.yaml                    # GPU456 실험 설정 파일
├── utils.py                               # 유틸리티 함수 (프롬프트 생성, 액션 파싱 등)
├── original_inference.py                  # 추론 및 평가 스크립트
├── training_unsloth.py                    # Unsloth를 사용한 고속 LoRA 파인튜닝
├── data_preprocessing.py                  # 데이터 전처리 스크립트
├── run_experiment.py                      # 실험 실행 스크립트
├── baseline_arc_results.json              # 베이스라인 ARC 결과
├── gpu456_inference_results.json          # GPU456 추론 결과
├── gpu456_integrated_prompt_results.json  # 통합 프롬프트 결과
├── gpu456_reproduction_results.json       # 재현 테스트 결과
├── experiment.log                         # 실험 로그
└── training.log                          # 학습 로그
```

## 🎯 실험 결과 요약

### 주요 성과
- **모델**: Llama-3.1-8B-Instruct + LoRA 어댑터
- **정확도**: 25% (5문제 중 1.25문제 정확 해결)
- **데이터**: ReARC 데이터셋 7개 문제에서 학습/평가
- **방법**: Few-shot learning + BARC 형식 프롬프트

### 성능 분석
| 문제 ID | 정확도 | Pixel 정확도 | 상태 |
|---------|--------|-------------|------|
| 4258a5f9 | 0.00 | 0.412 | ❌ |
| 445eab21 | 1.00 | 1.000 | ✅ |
| 6f8cd79b | 0.00 | 0.333 | ❌ |
| bb43febb | 0.00 | 0.361 | ❌ |
| bda2d7a6 | 0.25 | 0.543 | 🔸 |

## 🚀 사용 방법

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install torch transformers peft unsloth trl datasets numpy tqdm pyyaml

# GPU 메모리 최소 16GB 필요 (float16 precision)
```

### 2. 모델 로딩 및 추론
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 베이스 모델 로딩
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 어댑터 로딩
model_path = "/opt/dlami/nvme/seungpil/models_gpu456/unsloth_lora_model"
model = PeftModel.from_pretrained(base_model, model_path)

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

### 3. 추론 실행
```python
from utils import create_inference_prompt, parse_action_sequence_from_llm

# 프롬프트 생성 (few-shot examples 포함)
prompt = create_inference_prompt(
    input_grid=test_input,
    train_examples=few_shot_examples,
    use_barc_format=True
)

# 모델 추론
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(
    inputs['input_ids'],
    max_new_tokens=50,
    temperature=0.1,
    do_sample=True
)

# 액션 시퀀스 파싱
response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):])
actions = parse_action_sequence_from_llm(response)
```

### 4. 전체 평가 실행
```bash
cd final_model_result/
python original_inference.py --config config_ddp_456.yaml --model_path /opt/dlami/nvme/seungpil/models_gpu456/unsloth_lora_model
```

## 📊 결과 파일 설명

### 평가 결과
- **`gpu456_inference_results.json`**: 기본 추론 결과
- **`gpu456_integrated_prompt_results.json`**: 통합 프롬프트 사용 결과
- **`gpu456_reproduction_results.json`**: 재현 테스트 결과

### 로그 파일
- **`experiment.log`**: 전체 실험 과정 로그
- **`training.log`**: 모델 학습 과정 로그

### 설정 파일
- **`config_ddp_456.yaml`**: GPU456 실험의 모든 하이퍼파라미터와 설정

## 🔬 기술적 세부사항

### 모델 아키텍처
- **베이스 모델**: meta-llama/Llama-3.1-8B-Instruct
- **파인튜닝**: LoRA (Low-Rank Adaptation)
- **프롬프트 형식**: BARC (색상 코딩) + Few-shot learning

### 액션 시퀀스
모델은 다음 액션들을 학습합니다:
- `left_rotate` (0): 90도 반시계방향 회전
- `right_rotate` (1): 90도 시계방향 회전
- `horizontal_flip` (2): 수평 뒤집기
- `vertical_flip` (3): 수직 뒤집기
- `submit` (4): 최종 제출

### 프롬프트 예시
```
Problem: Transform the input grid by applying the correct sequence of transformations.

Examples:
Input:
🟦🟦🟥
🟦🟥🟦
🟥🟦🟦

Actions: [right_rotate,submit]

Now solve:
Input:
🟥🟦🟦
🟦🟦🟥
🟦🟥🟦

Actions:
```

## ✅ 재현 가능성

**완전 재현 가능**: 이 폴더의 파일들만으로 GPU456 모델의 모든 기능을 재현할 수 있습니다.

상세한 재현 테스트 결과는 `REPRODUCTION_TEST.md`를 참조하세요.

### 필요사항
1. 베이스 모델: `meta-llama/Llama-3.1-8B-Instruct` (HuggingFace에서 다운로드)
2. LoRA 어댑터: `/opt/dlami/nvme/seungpil/models_gpu456/unsloth_lora_model/`
3. GPU 메모리: 최소 16GB

## 📈 성능 개선 가능성

### 현재 한계
- 복잡한 변환 시퀀스에서 어려움
- 색상 패턴 인식 부정확
- 큰 그리드에서 성능 저하

### 개선 방안
- 더 많은 학습 데이터
- 더 복잡한 프롬프트 엔지니어링
- 멀티스텝 추론 파이프라인
- 앙상블 방법

## 📞 연락처

문의사항이나 재현에 어려움이 있으시면 프로젝트 관리자에게 연락하세요.