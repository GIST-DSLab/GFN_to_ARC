# 🔍 Multi-GPU Training Monitoring Guide

## 🎯 현재 실행 중인 학습

### GPU 분배:
- **GPU 4**: Problem 86, 139 (tmux session: `gpu4`)
- **GPU 5**: Problem 149, 154 (tmux session: `gpu5`) 
- **GPU 6**: Problem 178, 240 (tmux session: `gpu6`)
- **GPU 7**: Problem 379 (tmux session: `gpu7`)

## 📊 실시간 모니터링 명령어

### 1. tmux 세션 연결:
```bash
# GPU 4 세션 연결
tmux attach-session -t gpu4

# GPU 5 세션 연결  
tmux attach-session -t gpu5

# GPU 6 세션 연결
tmux attach-session -t gpu6

# GPU 7 세션 연결
tmux attach-session -t gpu7

# 세션에서 나가기: Ctrl+B, D
```

### 2. 빠른 상태 확인:
```bash
# 모든 tmux 세션 목록
tmux list-sessions

# 각 세션의 현재 출력 확인
tmux capture-pane -t gpu4 -p
tmux capture-pane -t gpu5 -p
tmux capture-pane -t gpu6 -p  
tmux capture-pane -t gpu7 -p

# GPU 사용률 확인
nvidia-smi

# 진행 중인 프로세스 확인
ps aux | grep python
```

### 3. 로그 파일 모니터링:
```bash
# 각 GPU별 로그 실시간 확인
tail -f /opt/dlami/nvme/seungpil/results_gpu4/training.log
tail -f /opt/dlami/nvme/seungpil/results_gpu5/training.log
tail -f /opt/dlami/nvme/seungpil/results_gpu6/training.log
tail -f /opt/dlami/nvme/seungpil/results_gpu7/training.log
```

### 4. 모델 저장 경로:
- GPU 4: `/opt/dlami/nvme/seungpil/models_gpu4/`
- GPU 5: `/opt/dlami/nvme/seungpil/models_gpu5/`
- GPU 6: `/opt/dlami/nvme/seungpil/models_gpu6/`
- GPU 7: `/opt/dlami/nvme/seungpil/models_gpu7/`

## ⚠️ 문제 해결

### 세션 종료 시:
```bash
# 모든 세션 종료
tmux kill-session -t gpu4
tmux kill-session -t gpu5
tmux kill-session -t gpu6
tmux kill-session -t gpu7

# 다시 시작
./run_gpu4.sh &
./run_gpu5.sh &
./run_gpu6.sh &
./run_gpu7.sh &
```

## 🚀 현재 상태
- 모든 GPU가 데이터 전처리 단계 완료 후 학습 시작 예정
- 각 GPU별로 독립적인 단일 GPU 학습 진행
- 메모리 최적화 설정 적용됨