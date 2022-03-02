import numpy as np
import torch
from torch import  nn
import os
from maddpg.maddpg import MADDPG
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple
from common.utils import OUNoise,GaussianNoise
class Agent(nn.Module):
    def __init__(self, agent_id, args):
        super().__init__()
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)  
        #self.noise = OUNoise(self.args.action_shape[self.agent_id], args.seed)
        self.noise = GaussianNoise(self.args.action_shape[0], seed=45)

    def select_action(self, o, noise_rate, epsilon):
        #if np.random.uniform() < epsilon or 1:
        if 1:
        #     a= np.random.uniform(-1,1, self.args.action_shape[self.agent_id])
        # else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(self.args.device)
            self.policy.actor_network.eval()
            with torch.no_grad():        
                pi = self.policy.actor_network(inputs).squeeze(0).detach().to(self.args.device)
            self.policy.actor_network.train()
            if noise_rate !=0:
                pi = pi + self.noise.sample()
            a=pi.cpu().numpy()
            # if noise_rate !=0:
            #    a += self.noise.sample()
            # noise = noise_rate * self.args.high_action * np.random.randn(*a.shape)  # gaussian noise
            # a += noise
            a = np.clip(a, -1, 1)
            #a=torch.softmax(torch.tensor(a).float().to(self.args.device),dim=0).cpu().numpy()
            # a[self.args.continuous_action_space:]=torch.softmax(torch.tensor(a[self.args.continuous_action_space:]).float().to(self.args.device),dim=0).cpu().numpy()
            #a[:self.args.continuous_action_space]=self.args.high_action*torch.softmax(torch.tensor(a[:self.args.continuous_action_space]).float().to(self.args.device)).cpu().numpy()
        # c_a=torch.softmax(torch.tensor(a[:self.args.continuous_action_space]).float(),dim=0)
        # a[:self.args.continuous_action_space]=c_a.cpu().numpy()
        #d_a=torch.argmax(torch.softmax(torch.tensor(a[self.args.continuous_action_space:]).float(),dim=0))
        # d_a=torch.sigmoid(torch.tensor(a[self.args.continuous_action_space:]).float().to(self.args.device))
        # d_a=d_a.cpu().numpy()
        # d_a=np.random.binomial(1,d_a)
        # for i in range(self.args.discrete_action_space):
        #     if i==d_a:
        #         a[int(self.args.continuous_action_space+d_a)]=1
        #     else:
        #         a[int(self.args.continuous_action_space+i)]=0
        #a[:self.args.continuous_action_space]= a[:self.args.continuous_action_space]
       # a[2]=0.5*a[2]
        #a[self.args.continuous_action_space:]=np.random.binomial(1,np.abs(a[self.args.continuous_action_space:]))
        a[self.args.continuous_action_space:]=(np.abs(a[self.args.continuous_action_space:])>0.3)*1
        # if(self.agent_id==0):
        #     print(a)
        return a.copy()


    def learn(self, experiences, other_agents):
        self.policy.train(experiences, other_agents)

    def get_actor_params(self):
        return self.policy.get_actor_params()

    def load_actor_params(self, state_dict):
        self.policy.actor_network.load_state_dict(state_dict)

