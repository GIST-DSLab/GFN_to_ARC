import torch

def get_device(cuda_num: int):
    """CUDA 장치 번호에 따라 Device를 설정."""
    if torch.cuda.is_available() and cuda_num >= 0:
        return torch.device(f"cuda:{cuda_num}")
    return torch.device("cpu")

CONFIG = {
    # Device 설정 (CUDA 사용 여부 및 장치 번호)
    "CUDANUM": 0,  # 사용할 CUDA 장치 번호 (-1: CPU)
    "DEVICE": get_device(0),  # 위에서 정의한 `get_device` 함수로 설정
    
    # 학습 및 환경 설정
    "TASKNUM": 178,  # ARC Task 번호
    "ACTIONNUM": 5,  # 가능한 액션의 수
    "EP_LEN": 10,  # 에피소드 길이
    "ENV_MODE": "entire",  # 환경 모드 ("entire", "partial")
    "NUM_EPOCHS": 3,  # 학습 에포크 수
    "BATCH_SIZE": 1,  # 배치 크기

    # 손실 함수 설정
    "LOSS_METHOD": "trajectory_balance_loss",  # 사용할 손실 함수 ("trajectory_balance_loss", "detailed_balance_loss", "subtb_loss")

    # Weights & Biases 사용 여부
    "WANDB_USE": True,  # W&B 로깅 활성화 여부
    "FILENAME": "geometric_10,5_taskg_rscale10_178",  # 로그 파일 이름

    # Replay Buffer 설정
    "REPLAY_BUFFER_CAPACITY": 10000,  # Replay Buffer 최대 용량

    # 보상 임계값 설정
    "REWARD_THRESHOLD_INIT": 1.0,  # 초기 보상 임계값
    "REWARD_THRESHOLD_MAX": 10.0,  # 최대 보상 임계값
    "REWARD_THRESHOLD_INCR_RATE": 0.01,  # 보상 임계값 증가율

    # Off-policy 학습 사용 여부
    "USE_OFFPOLICY": False,  # Off-policy 학습 활성화 여부

    # 평가 관련 설정
    "EVAL_SAMPLES": 100,  # 평가 샘플 수
}
