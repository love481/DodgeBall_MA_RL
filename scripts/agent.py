import numpy as np
import torch
from torch import  nn
import os
from maddpg.maddpg import MADDPG
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple
class Agent(nn.Module):
    def __init__(self, agent_id, args):
        super().__init__()
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            a= np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)            
            pi = self.policy.actor_network(inputs).squeeze(0)
            a=pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*a.shape)  # gaussian noise
            a += noise
            a = np.clip(a, -self.args.high_action, self.args.high_action)

        # c_a=torch.softmax(torch.tensor(a[:self.args.continuous_action_space]).float(),dim=0)
        # a[:self.args.continuous_action_space]=c_a.cpu().numpy()
        d_a=torch.argmax(torch.softmax(torch.tensor(a[self.args.continuous_action_space:]).float(),dim=0))
        d_a=d_a.cpu().numpy()
        for i in range(self.args.discrete_action_space):
            if i==d_a:
                a[int(self.args.continuous_action_space+d_a)]=1
            else:
                a[int(self.args.continuous_action_space+i)]=0
        return a.copy()


    def learn(self, experiences, other_agents):
        self.policy.train( experiences, other_agents)

