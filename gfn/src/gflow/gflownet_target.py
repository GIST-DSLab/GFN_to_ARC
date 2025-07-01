import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Geometric, OneHotCategorical

import numpy as np
from .log import Log
from collections import OrderedDict


class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy=None, total_flow=None, env=None, device='cuda', env_style="point", num_actions=12, ep_len=10):
        super().__init__()

        if total_flow is None:
            self.total_flow = nn.Parameter(torch.tensor(1.0).to(device))
        else:
            self.total_flow = total_flow  # Log Z 

        if backward_policy is not None:
            self.backward_policy = backward_policy.to(device)
        else:
            self.backward_policy = backward_policy

        self.forward_policy = forward_policy.to(device)
        self.env = env
        self.device = device
        self.actions = {}
        self.num_actions = num_actions
        self.max_length = ep_len
        self.env_style = env_style  # "point", "bbox", etc.

        self.mask = None
        self.emb_dag = None
        self.dag_s = None
        self.ac = None

    def set_env(self, env):
        self.env = env

    def forward_probs(self, s, mask, sample=True, action=None):
        if sample:
            probs, selection = self.forward_policy(s, mask)
            return probs, selection
        else:
            probs, selection = self.forward_policy(s, mask)
            l = int(probs.shape[1] / 2)

            logpf = self.logit_to_pf(probs[:, :l], sample=True, action=action)
            logpb = self.logit_to_pb(probs[:, l:])
            return logpf, logpb

    def logit_to_pf(self, logits, sample=True, action=None):
        if 0.0 in logits[0]:
            logits = logits.clone()
            logits[0] = logits[0] + 1e-20

        if sample:
            fwd_prob = Geometric(probs=logits[0])
            self.ac = fwd_prob.sample().argmin()
            fwd_prob_s = fwd_prob.log_prob(self.ac)[self.ac]
        else:
            fwd_prob = Geometric(probs=logits[0])
            self.ac = action
            fwd_prob_s = fwd_prob.log_prob(self.ac)

        if fwd_prob_s.dim() == 0:
            fwd_prob_s = fwd_prob_s.unsqueeze(0)
        return fwd_prob_s

    def logit_to_pb(self, logits):
        back_probs = logits
        back_probs_s = Categorical(logits=back_probs).log_prob(self.ac)
        if back_probs_s.dim() == 0:
            back_probs_s = back_probs_s.unsqueeze(0)
        return back_probs_s

    def sample_states(self, s0, info=None, return_log=True, batch_size=128, use_selection=False):
        if self.env is None:
            raise ValueError("Environment is not set. Please call set_env() before using this method.")

        iter = 0

        s = torch.tensor(s0["input"], dtype=torch.float).unsqueeze(0).to(self.device)
        answers = np.array(self.env.unwrapped.answer)
        t_ = torch.from_numpy(answers).unsqueeze(0).to(self.device)

        h, w = t_.shape[1], t_.shape[2]

        if use_selection:
            if self.mask is None or iter == 0:
                self.mask = torch.zeros((batch_size, 30, 30), dtype=torch.bool).to(self.device)
                self.mask[:, :h, :w] = True
        else:
            if batch_size > 1:
                self.mask = torch.zeros((batch_size, 30, 30), dtype=torch.bool).to(self.device)
                self.mask[:, :h, :w] = True
            else:
                self.mask = torch.zeros((30, 30), dtype=torch.bool).to(self.device)
                self.mask[:h, :w] = True

        log = Log(s[:, :3, :3], self.backward_policy, self.total_flow, self.env, tstate=s0, emb_s=None, num_actions=self.num_actions) if return_log else None
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        while not is_done.all():
            iter += 1

            if use_selection:
                active_mask = (~is_done).unsqueeze(1).unsqueeze(2)
                probs_s, selection = self.forward_probs(s * active_mask, self.mask.clone() * active_mask, iter)
                prob = Categorical(probs=probs_s)
                ac = prob.sample()
                actions = tuple([selection[i] for i in range(batch_size)] + [ac.cpu().numpy()])
            else:
                probs, selection = self.forward_probs(s, self.mask, sample=True)
                l = int(probs.shape[1] / 2)

                fwd_prob_s = self.logit_to_pf(probs[:, :l])
                back_probs_s = self.logit_to_pb(probs[:, l:])

                actions = {
                    "operation": self.ac.cpu().numpy(),
                    "selection": self.mask.cpu().numpy()
                }

            results = self.env.step(actions)
            states, rewards, dones, truncated, infos = results

            gamma = 0.9
            is_done = torch.tensor(dones, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device) * 15 * (gamma ** iter)

            s = torch.where(~is_done, torch.tensor(states['grid'], dtype=torch.float, device=self.device).unsqueeze(0), s)

            if use_selection:
                for i, sel in enumerate(selection):
                    if not is_done[i]:
                        self.mask[i, sel[0], sel[1]] = True

            if return_log:
                log.log(s=s.clone(), probs=fwd_prob_s, back_probs=back_probs_s, actions=self.ac, tstate=states, rewards=rewards, done=is_done)

            if (iter >= self.max_length) or is_done.all():
                break

        return s, log if return_log else s
