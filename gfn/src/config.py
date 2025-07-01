import torch

def get_device(cuda_num: int):
    """Set device based on CUDA device number."""
    if torch.cuda.is_available() and cuda_num >= 0:
        return torch.device(f"cuda:{cuda_num}")
    return torch.device("cpu")

CONFIG = {
    # Device configuration (CUDA availability and device number)
    "CUDANUM": 0,  # CUDA device number to use (-1: use CPU)
    "DEVICE": get_device(0),  # Set using the `get_device` function above

    # Training and environment settings
    "TASKNUM": 240,  # ARC Task number
    "SUBTASKNUM": 2,
    "ACTIONNUM": 5,  # Number of possible actions
    "EP_LEN": 10,  # Episode length
    "ENV_MODE": "entire",  # Environment mode ("entire", "partial")
    "NUM_EPOCHS": 1,  # Number of training epochs
    "BATCH_SIZE": 1,  # Batch size

    # Loss function settings
    "LOSS_METHOD": "trajectory_balance_loss",  # Loss function to use ("trajectory_balance_loss", "detailed_balance_loss", "subtb_loss")

    # Weights & Biases (wandb) logging
    "WANDB_USE": True,  # Enable W&B logging
    "FILENAME": "geometric_10,5_taskg_rscale10_178",  # Log file name

    # Replay buffer settings
    "REPLAY_BUFFER_CAPACITY": 10000,  # Maximum capacity of the replay buffer

    # Reward threshold settings
    "REWARD_THRESHOLD_INIT": 1.0,  # Initial reward threshold
    "REWARD_THRESHOLD_MAX": 10.0,  # Maximum reward threshold
    "REWARD_THRESHOLD_INCR_RATE": 0.01,  # Reward threshold increment rate

    # Off-policy training
    "USE_OFFPOLICY": False,  # Enable off-policy training

    # Evaluation settings
    "EVAL_SAMPLES": 100,  # Number of evaluation samples
}
