import torch

def MSE_reward(self, s, pad_terminal):
        """
        Returns the reward associated with a given state.

        Args:
            s: An NxD matrix representing N states
        """
        # s = self.normalize(s)
        # pad_terminal = self.normalize(pad_terminal)

        s = self.normalize(s)
        pad_terminal = self.normalize(pad_terminal)

        # r = torch.exp(-torch.norm(s - pad_terminal)**2 / scalefactor)

        r = ((pad_terminal.squeeze() - s.squeeze())**2 + 1e-6)
        r[torch.isinf(r)] = 0

        r = torch.mean(r) 
        r = torch.exp(-r)

        if r == 0 :
            r = torch.tensor(1e-10).to(self.device)

        return r
    
def MSE_energy_reward(self, s, pad_terminal):
    """
    Returns the reward associated with a given state.

    Args:
        s: An NxD matrix representing N states
    """
    # s = self.normalize(s)
    # pad_terminal = self.normalize(pad_terminal)

    # MSE 
    r = ((pad_terminal.squeeze() - s.squeeze())**2 + 1e-6) # pad_terminal is for ARC

    # if inf is detected, set it to 0
    r[torch.isinf(r)] = 0
    # for i in range(len(r)):
    #     for j in range(len(r[0])):
    #         if r[i][j] == float("inf"):
    #             r[i][j] = 0
    mse_reward = 1 / (r.sum() + 1) 

    # if the value is inf, set it to 0
    # if mse_reward == float("inf"):
    #     mse_reward = 10000

    return mse_reward*1000

def pixel_Reward(self, s, pad_terminal):

    reward = 0
    for i in range(len(s)):
        for j in range(len(s[0])):
            if s[i][j] == pad_terminal[i][j]:
                reward += 1
    
    return torch.tensor(reward, dtype = torch.float32).to(self.device)

def human_reward(self):
    r = int(input("input reward : "))
    reward = torch.tensor(r, dtype = torch.float32).to(self.device)

    return reward
def task_specific_reward(self, s, answer, i):
    reward = 0
    gray_positions = [(x, y) for x in range(answer.shape[0])
                    for y in range(answer.shape[1]) if answer[x, y] == 5]
    black_positions = [(x, y) for x in range(answer.shape[0])
                    for y in range(answer.shape[1]) if answer[x, y] == 0]

    positions_dict = {
        0: [(3, 2), (4, 2), (3, 3), (4, 3), (10, 3)],
        1: [(2,3), (9,1), (9,2), (10,2), (10,1), (4,7), (5,7), (6,7), (7,7), (4,8), (5,8), (6,8), (7,8), (4,9), (5,9), (6,9), (7,9), (4,10), (5,10), (6,10), (7,10)],
        2: [(4,8), (4,9), (4,8), (5,9)],
        3: []
    }

    positions_to_check = positions_dict.get(i, [])

    for pos in positions_to_check:
        if s[pos] == 2:
            reward += 1000
        elif s[pos] == 0:
            reward += 10

    for x in range(s.shape[0]):
        for y in range(s.shape[1]):
            if (x, y) not in positions_to_check and s[x, y] == 2:
                reward += 1

    for pos in gray_positions:
        if s[pos] == 0:
            reward += 1
        if s[pos] == 5:
            reward += 10
    for pos in black_positions:
        if s[pos] == 0 :
            reward +=10

    return torch.tensor(reward, dtype = torch.float32).to(self.device)

def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())