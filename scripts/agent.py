import numpy as np
import torch
from torch import  nn
import os
from maddpg.maddpg import MADDPG
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple
from common.utils import OUNoise,GaussianNoise
import random
class Agent(nn.Module):
    def __init__(self, agent_id, args):
        super().__init__()
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)  
        #self.noise = OUNoise(self.args.action_shape[0], args.seed)
        self.noise = GaussianNoise(self.args.action_shape[0], seed=8)
        #self.seed = random.seed(89)

    def select_action(self, o, noise_rate, epsilon):
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(self.args.device)
        self.policy.actor_network.eval()
        with torch.no_grad():        
            pi = self.policy.actor_network(inputs).squeeze(0).detach().to(self.args.device)
        self.policy.actor_network.train()
        if noise_rate !=0:
            pi = pi + self.noise.sample()
        a=pi.cpu().numpy()
        a = np.clip(a, -1, 1)
        a[self.args.continuous_action_space:]=(np.abs(a[self.args.continuous_action_space:])>0.3)*1
        return a.copy()


    def learn(self, experiences, other_agents):
        self.policy.train(experiences, other_agents)

    def get_actor_params(self):
        return self.policy.get_actor_params()

    def load_actor_params(self, state_dict):
        self.policy.actor_network.load_state_dict(state_dict)

