import numpy as np
import torch
from torch import  nn
import os
from maddpg.maddpg import MADDPG
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple
class Agent(nn.Module):
    def __init__(self, agent_id, args,team_id):
        super().__init__()
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id,team_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.abs(np.random.uniform()) < epsilon:
            a= np.random.uniform(-1, 1, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(self.args.device)            
            pi = self.policy.actor_network(inputs).squeeze(0)
            a=pi.cpu().numpy()
            noise = noise_rate *  np.random.randn(*a.shape)  # gaussian noise
            a += noise
            # a = np.clip(a, -self.args.high_action, self.args.high_action)
            a[self.args.continuous_action_space:]=torch.sigmoid(torch.tensor(a[self.args.continuous_action_space:]).float().to(self.args.device)).cpu().numpy()
            a[:self.args.continuous_action_space]=self.args.high_action*torch.tanh(torch.tensor(a[:self.args.continuous_action_space]).float().to(self.args.device)).cpu().numpy()

        # c_a=torch.softmax(torch.tensor(a[:self.args.continuous_action_space]).float(),dim=0)
        # a[:self.args.continuous_action_space]=c_a.cpu().numpy()
        # d_a=torch.argmax(torch.softmax(torch.tensor(a[self.args.continuous_action_space:]).float(),dim=0))
        # d_a=torch.sigmoid(torch.tensor(a[self.args.continuous_action_space:]).float().to(self.args.device))
        # d_a=d_a.cpu().numpy()
        # d_a=np.random.binomial(1,d_a)
        # for i in range(self.args.discrete_action_space):
        #     if i==d_a:
        #         a[int(self.args.continuous_action_space+d_a)]=1
        #     else:
        #         a[int(self.args.continuous_action_space+i)]=0
        a[:self.args.continuous_action_space]=self.args.high_action*a[:self.args.continuous_action_space]
        a[2]=0.05*a[2]
        a[self.args.continuous_action_space:]=np.random.binomial(1,np.abs(a[self.args.continuous_action_space:]))
        return a.copy()


    def learn(self, experiences, other_agents,i):
        self.policy.train(experiences, other_agents,i)

