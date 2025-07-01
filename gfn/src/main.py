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
    parser.add_argument("--subtask_num", type=int, default=CONFIG["SUBTASKNUM"])
    return parser.parse_args()

def main():
    args = parse_arguments()
    seed_everything(777)
    device = CONFIG["DEVICE"]
    use_offpolicy = CONFIG["USE_OFFPOLICY"]

    if CONFIG["WANDB_USE"]:
        setup_wandb(
            project_name="gflow_research",
            entity="hsh6449",
            config={
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "env_mode": args.env_mode,
                "prob_index": args.prob_index,
                "num_actions": args.num_actions,
                "use_offpolicy": args.use_offpolicy
            },
            run_name=CONFIG["FILENAME"]
        )
    
    if args.save_trajectories:
        save_gflownet_trajectories(args.num_trajectories, args.save_trajectories, args)
    else:
       model, _ = train_model(
        num_epochs= CONFIG["NUM_EPOCHS"],
        batch_size=CONFIG["BATCH_SIZE"],
        device=device,
        env_mode=CONFIG["ENV_MODE"],
        prob_index=CONFIG["TASKNUM"],
        num_actions=CONFIG["ACTIONNUM"],
        args=args,
        use_offpolicy=use_offpolicy,
        sub_task = CONFIG["SUBTASKNUM"]
    )
if __name__ == "__main__":
    main()
