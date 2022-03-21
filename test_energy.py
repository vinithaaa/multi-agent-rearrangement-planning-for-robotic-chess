import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

class FuncMinGTEnv():
    def __init__(self, batch_size, input_size, batched_func):
        self.a_size = input_size
        self.func = batched_func
        self.state = None
        self.reset_state(batch_size)

    def reset_state(self, batch_size):
        self.B = batch_size

    def rollout(self, actions):
        # Uncoditional action sequence rollout
        # TxBxA
        T = actions.size(0)
        total_r = torch.zeros(self.B, requires_grad=True, device=actions.device)
        for i in range(T):
            _, r, done = self.step(actions[i])
            total_r = total_r + r
            if(done):
                break
        return total_r

    def step(self, action):
        self.state = self.sim(self.state, action)
        o = self.calc_obs(self.state)
        r = self.calc_reward(self.state, action)
        # always done after first step
        return o, r, True

    def sim(self, state, action):
        return state

    def calc_obs(self, state):
        return None

    def calc_reward(self, state, action):
        return -self.func(action)
