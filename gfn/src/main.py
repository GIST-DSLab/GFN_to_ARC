import torch
import argparse
from config import CONFIG
from gflow.utils import seed_everything, setup_wandb
from train import train_model, save_gflownet_trajectories


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"])
    parser.add_argument("--num_epochs", type=int, default=CONFIG["NUM_EPOCHS"])
    parser.add_argument("--env_mode", type=str, default=CONFIG["ENV_MODE"])
    parser.add_argument("--prob_index", type=int, default=CONFIG["TASKNUM"])
    parser.add_argument("--num_actions", type=int, default=CONFIG["ACTIONNUM"])
    parser.add_argument("--ep_len", type=int, default=CONFIG["EP_LEN"])
    parser.add_argument("--device", type=int, default=CONFIG["CUDANUM"])
    parser.add_argument("--use_offpolicy", action="store_true", default=False)
    parser.add_argument("--sampling_method", type=str, default="prt", 
                        choices=["prt", "fixed_ratio", "egreedy"])
    parser.add_argument("--save_trajectories", type=str, default=None)
    parser.add_argument("--num_trajectories", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_arguments()
    seed_everything(48)
    device = CONFIG["DEVICE"]
    use_offpolicy = CONFIG["USE_OFFPOLICY"]

    if CONFIG["WANDB_USE"]:
        setup_wandb(
            project_name="gflow_research",
            entity="hsh6449",
            config={
                "num_epochs": args.num_epochs,
                "batch_size": batch_size,
                "env_mode": env_mode,
                "prob_index": prob_index,
                "num_actions": num_actions,
                "use_offpolicy": use_offpolicy
            },
            run_name=CONFIG["FILENAME"]
        )
    
    if args.save_trajectories:
        save_gflownet_trajectories(args.num_trajectories, args.save_trajectories, args)
    else:
       model, env = train_model(
        num_epochs=10,
        batch_size=32,
        device=device,
        env_mode="entire",
        prob_index=CONFIG["TASKNUM"],
        num_actions=CONFIG["ACTIONNUM"],
        args=args,
        use_offpolicy=use_offpolicy
    )
if __name__ == "__main__":
    main()
