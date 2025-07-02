#!/bin/bash

# GPU6 학습을 위한 안정적인 tmux 세션 스크립트

SESSION_NAME="gpu6"
WORK_DIR="/home/ubuntu/GFN_to_ARC/gfn/src"

echo "Creating tmux session: $SESSION_NAME"

# 기존 세션이 있으면 종료
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Killing existing session: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
fi

# 새 tmux 세션 생성 (detached 모드)
tmux new-session -d -s $SESSION_NAME

# 작업 디렉토리로 이동
tmux send-keys -t $SESSION_NAME "cd $WORK_DIR" Enter

# 환경 변수 설정 (GPU 6 사용)
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=5,6" Enter
tmux send-keys -t $SESSION_NAME "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter
tmux send-keys -t $SESSION_NAME "export CUDA_LAUNCH_BLOCKING=0" Enter

# 안정적인 학습 실행 (오류 발생 시에도 세션 유지)
tmux send-keys -t $SESSION_NAME "echo 'Starting robust GFN ARC training...'" Enter
tmux send-keys -t $SESSION_NAME "python robust_training.py" Enter

echo "✅ tmux session '$SESSION_NAME' created successfully!"
echo "📊 To monitor progress: tmux attach -t $SESSION_NAME"
echo "🔄 To check if session is running: tmux list-sessions | grep $SESSION_NAME"
echo "📝 To view session output without attaching: tmux capture-pane -t $SESSION_NAME -p"

# 세션 상태 확인
sleep 2
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "✅ Session is running successfully"
    tmux list-sessions | grep $SESSION_NAME
else
    echo "❌ Failed to create session"
fi