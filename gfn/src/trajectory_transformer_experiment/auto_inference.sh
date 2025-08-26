#\!/bin/bash

echo "Waiting for training to complete..."

# 학습 완료 대기 (training.py 프로세스 종료 대기)
while pgrep -f "python training.py" > /dev/null; do
    echo "$(date): Training still running..."
    sleep 30
done

echo "Training completed\! Starting inference..."

# 추론 실행
CUDA_VISIBLE_DEVICES=5 python inference.py --config configs/config.yaml --model_path models/checkpoint_best.pt

echo "Inference completed\!"

# 결과 확인
if [ -f "results/evaluation_results.json" ]; then
    echo "=== EVALUATION RESULTS ==="
    python -c "
import json
with open('results/evaluation_results.json', 'r') as f:
    results = json.load(f)
print(f'Overall Accuracy: {results[\"overall_accuracy\"]:.1%}')
print(f'Total Tests: {results[\"total_tests\"]}')
print(f'Correct: {results[\"total_correct\"]}')
"
fi
