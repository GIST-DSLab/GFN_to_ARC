# 🚀 ARC LLM Training Guide

## ✅ 문제 해결 완료

### 발생했던 문제들:
1. **단일 GPU OOM (Out of Memory)** - ✅ 해결됨
2. **멀티 GPU NCCL 타임아웃** - ⚠️ 이 환경에서는 불안정

### 적용된 해결책:

#### 단일 GPU 최적화:
- **배치 크기**: 4 → 1
- **그래디언트 축적**: 2 → 8 (효과적 배치 크기 유지)
- **평가 비활성화**: `eval_strategy="no"`
- **메모리 최적화**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

#### 설정 파일 (`configs/config.yaml`):
```yaml
batch_size: 1
gradient_accumulation_steps: 8
```

## 🎯 권장 실행 방법

### 1. 단일 GPU (권장):
```bash
./run_single_gpu.sh
```

또는 직접:
```bash
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python training.py
```

### 2. 멀티 GPU (불안정):
현재 환경에서는 NCCL 통신 문제로 권장하지 않음

## 📊 성능 확인

- **단일 GPU**: ✅ 정상 작동 (약 3.9 it/s)
- **메모리 사용량**: 최적화됨
- **학습 진행**: 정상 (평가 없이 훈련만 수행)

## 🔍 모니터링

학습 진행 상황은 다음에서 확인:
- **로그**: `./results/training.log`
- **WandB**: 자동으로 생성된 링크 확인
- **모델 저장**: `./models/arc_action_model_DialoGPT-medium/`

## 💡 추가 팁

1. **더 큰 모델 사용 시**: 배치 크기를 더 줄이고 그래디언트 축적을 늘리세요
2. **메모리 부족 시**: `max_length`를 줄여보세요
3. **학습 속도 향상**: 더 강력한 GPU나 여러 개의 단일 GPU 인스턴스 사용