from importlib.resources import path
from pickletools import pybool
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pybullet
import math

def get_test_energy2d_env(batch_size, rob, path):
    return FuncMinGTEnv(batch_size, 2, test_energy2d, rob=rob, path=path)

def test_energy2d(action_batch):
    assert action_batch.dim() == 2
    assert action_batch.size()[1] == 2

    opt_point = torch.tensor([[1.0, 1.0]],
                    requires_grad=False,
                    device=action_batch.device)

    return ((action_batch-opt_point)**2).sum(-1)

def get_test_energy(opt_point):
    def test_energy(query_batch):
        assert query_batch.dim() == 2
        assert query_batch.size(1) == opt_point.size(-1)

        return ((query_batch-opt_point)**2).sum(-1)
    return test_energy


class FuncMinGTEnv():
    def __init__(self, batch_size, input_size, batched_func, rob, path):
        self.a_size = input_size
        self.func = batched_func
        self.state = None
        self.rob = rob
        self.path = path
        self.numInPath = 0
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
        self.numInPath += 1
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
        x = pybullet.getJointInfo(self.rob, 0)[2]
        y = pybullet.getJointInfo(self.rob, 0)[3]
        z = pybullet.getJointInfo(self.rob, 0)[4]
        x2 = path[self.numInPath][0]
        y2 = path[self.numInPath][1]
        z2 = path[self.numInPath][2]
        distance = math.dist((x, y, z), (x2, y2, z2))
        return distance
