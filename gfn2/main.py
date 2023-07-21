import torch
import numpy as np
from tqdm import tqdm

from gfn import LogitPBEstimator, LogitPFEstimator, LogZEstimator
from gfn.envs import HyperGrid
from gfn.losses import TBParametrization, TrajectoryBalance
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler

from ARC.ARCenv import MiniArcEnv

class TestEnv1:
    INITIAL = np.array(
        [[0,4,4,0,0],
         [0,4,6,6,0],
         [0,4,4,6,0],
         [0,4,4,0,0],
         [0,6,4,0,0]])
    GOAL = np.array(
        [[0,6,6,0,0],
         [0,6,4,4,0],
         [0,6,6,4,0],
         [0,6,6,0,0],
         [0,4,6,0,0]])
    OBJ_MAP = np.array(
        [[0,1,1,0,0],
         [0,1,2,2,0],
         [0,1,1,2,0],
         [0,1,1,0,0],
         [0,3,1,0,0]])

    @classmethod
    def get_args(cls):
        return cls.INITIAL.copy(), cls.GOAL.copy(), cls.OBJ_MAP.copy()
    

if __name__ == "__main__":

    env = MiniArcEnv(*TestEnv1.get_args())

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