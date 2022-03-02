import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


#todo: modify the class to receive hidden layer

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,args, agent_id, fc1_units=200, fc2_units=5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.max_action = args.high_action
        self.agent_id=agent_id
        self.bn1 = nn.BatchNorm1d(args.obs_shape[0])
        self.fc1 = nn.Linear(args.obs_shape[0], fc1_units)
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.action_out_cont = nn.Linear(fc2_units, args.continuous_action_space)
        self.action_out_disc = nn.Linear(fc2_units, args.discrete_action_space)
        self.reset_parameters()
        self.to(args.device)

    def reset_parameters(self):
        # see DDPG paper chapter 7. for detail
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))  # * is for unpacking list or tuple
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.action_out_cont.weight.data.uniform_(-3e-3, 3e-3)
        self.action_out_disc.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        state = self.bn1(state)
        x = F.relu(self.fc1(state))
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        actions_con = torch.tanh(self.action_out_cont(x))
        #actions_con.detach().numpy()[:,:2]= self.max_action*actions_con.detach().numpy()[:,:2]
        actions_disc =torch.softmax(self.action_out_disc(x),dim=0)
        # if self.agent_id==0:
        #     print(torch.cat((actions_con, actions_disc),dim=1).cpu().numpy())
        return torch.cat((actions_con, actions_disc),dim=1)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self,args,fcs1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super().__init__()
        self.max_action = args.high_action
        self.bn1 = nn.BatchNorm1d(sum(args.obs_shape))
        self.fcs1 = nn.Linear(sum(args.obs_shape), fcs1_units)
        self.bn2 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + sum(args.action_shape), fc2_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        self.to(args.device)

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = torch.cat(state, dim=1)
        state=self.bn1(state)
        xs = F.relu(self.fcs1(state))
        xs = self.bn2(xs)
        action = torch.cat(action, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        x = torch.cat([xs, action], dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)
# define the actor network
# class Actor(nn.Module):
#     def __init__(self, args, agent_id):
#         super(Actor, self).__init__()
#         self.max_action = args.high_action
#         self.fc1 = nn.Linear(args.obs_shape[agent_id], 500)
#         self.fc2 = nn.Linear(500, 5)
#         self.action_out_cont = nn.Linear(5, args.continuous_action_space)
#         self.action_out_disc = nn.Linear(5, args.discrete_action_space)
 
#         self.to(args.device)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         actions_con = torch.tanh(self.action_out_cont(x))
#         actions_disc =torch.sigmoid(self.action_out_disc(x))
#         return torch.cat((actions_con, actions_disc),dim=1)


# class Critic(nn.Module):
#     def __init__(self, args):
#         super(Critic, self).__init__()
#         self.max_action = args.high_action
#         self.fc1 = nn.Linear(sum(args.obs_shape)+sum(args.action_shape), 1500)
#         self.fc2= nn.Linear(1500, 500)
#         self.q_out = nn.Linear(500, 1)
#         self.to(args.device)

#     def forward(self, state, action):
#         state = torch.cat(state, dim=1)
#         for i in range(len(action)):
#             action[i] /= self.max_action
#         action = torch.cat(action, dim=1)
#         x = torch.cat([state, action], dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         q_value =self.q_out(x)
#         return q_value
