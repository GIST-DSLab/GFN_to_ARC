import torch
from tqdm import tqdm

from gfn import LogitPBEstimator, LogitPFEstimator, LogZEstimator
from gfn.envs import HyperGrid
from gfn.losses import TBParametrization, TrajectoryBalance
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler

if __name__ == "__main__":

    env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8

    logit_PF = LogitPFEstimator(env=env, module_name="NeuralNet")
    logit_PB = LogitPBEstimator(
        env=env,
        module_name="NeuralNet",
        torso=logit_PF.module.torso,  # To share parameters between PF and PB
    )
    logZ = LogZEstimator(torch.tensor(0.0))

    parametrization = TBParametrization(logit_PF, logit_PB, logZ)

    actions_sampler = DiscreteActionsSampler(estimator=logit_PF)
    trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=actions_sampler)

    loss_fn = TrajectoryBalance(parametrization=parametrization)

    params = [
        {
            "params": [
                val for key, val in parametrization.parameters.items() if "logZ" not in key
            ],
            "lr": 0.001,
        },
        {"params": [val for key, val in parametrization.parameters.items() if "logZ" in key], "lr": 0.1},
    ]
    optimizer = torch.optim.Adam(params)

    for i in (pbar := tqdm(range(1000))):
        trajectories = trajectories_sampler.sample(n_trajectories=16)
        optimizer.zero_grad()
        loss = loss_fn(trajectories)
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            pbar.set_postfix({"loss": loss.item()})