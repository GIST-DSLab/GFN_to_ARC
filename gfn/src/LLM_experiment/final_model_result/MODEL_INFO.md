# GPU456 모델 정보

## 모델 경로
```
/opt/dlami/nvme/seungpil/models_gpu456/unsloth_lora_model/
```

## 모델 구성
- **베이스 모델**: `meta-llama/Llama-3.1-8B-Instruct`
- **파인튜닝 방법**: Unsloth LoRA (Low-Rank Adaptation)
- **LoRA 파라미터**: r=16, alpha=32, dropout=0.1
- **어댑터 크기**: ~160MB (`adapter_model.safetensors`)

## 학습 결과
- **정확도**: 25% (2/8 테스트 케이스)
- **성공한 문제**: Task 139 (6150a2bd), Task 178 (74dd1130)
- **학습 데이터**: 7개 ARC 문제, 각각 ~10k trajectory samples

## 모델 파일들
```
adapter_config.json          - LoRA 설정
adapter_model.safetensors    - LoRA 가중치 (160MB)
tokenizer.json              - 토크나이저 (17MB)
tokenizer_config.json       - 토크나이저 설정
chat_template.jinja         - Llama-3.1 채팅 템플릿
special_tokens_map.json     - 특수 토큰 매핑
training_args.bin          - 학습 인자
README.md                  - 모델 설명
```

## 재현 방법
1. 베이스 모델 다운로드: `meta-llama/Llama-3.1-8B-Instruct`
2. LoRA 어댑터 로드: `PeftModel.from_pretrained(base_model, model_path)`
3. 추론 실행: `reproduce_gpu456_inference.py` 사용

## 프롬프트 형식
- **Few-shot learning**: ARC training examples 최대 2개 사용
- **컬러 코딩**: 숫자 대신 색상 이름 (Black, Blue, Red, etc.)
- **Llama-3.1 chat format**: `<|begin_of_text|>...<|start_header_id|>...<|eot_id|>`
- **액션 시퀀스**: `[left_rotate, right_rotate, horizontal_flip, vertical_flip, submit]`