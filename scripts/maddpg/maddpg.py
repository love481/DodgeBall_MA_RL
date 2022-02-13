import torch
import os
from maddpg.actor_critic import Actor, Critic
from common.replay_buffer import Buffer
import torch.nn.functional as F
import csv
class MADDPG:
    def __init__(self, args, agent_id):  
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id)
        
        self.critic_network = Critic(args)
        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)
        self.huber_loss = torch.nn.SmoothL1Loss()
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_(param.data)
        # self.f = open("rewards.txt",'w')
        # self.writer = csv.writer(self.f)

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
    # update the network
    def train(self, experiences, other_agents):
        for key in experiences.keys():
            experiences[key] = torch.tensor(experiences[key], dtype=torch.float32).to(self.args.device)
        r = experiences['r_%d' % self.agent_id] 
        done=experiences['done_%d' % self.agent_id]
        o, a,o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(experiences['o_%d' % agent_id])
            a.append(experiences['a_%d' % agent_id])
            o_next.append(experiences['o_next_%d' % agent_id])

        # calculate the target Q value function
        a_next = []
        a_current=[]
        with torch.no_grad():
           
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    a_next.append(self.actor_target_network.forward(o_next[agent_id]))
                    #a_current.append(self.actor_network.forward(o[agent_id]))
                else:
                    a_next.append(other_agents[index].policy.actor_target_network.forward(o_next[agent_id]))
                    #a_current.append(other_agents[index].policy.actor_network.forward(o[agent_id]))
                    index += 1
            q_next = self.critic_target_network.forward(o_next, a_next)
            target_q = (r.unsqueeze(1) + (self.args.gamma *q_next*(1-done))).detach()

        # the q loss
        q_value = self.critic_network.forward(o, a)
        critic_loss =self.huber_loss(q_value,target_q)
        # if self.agent_id==0:
        #     print("critic loss for agent {} is {}".format(self.agent_id,critic_loss ))
        # the actor loss
        a[self.agent_id]=self.actor_network.forward(o[self.agent_id])
        actor_loss = - self.critic_network(o, a).mean()
        #if self.agent_id==0:
            #print(" actor_lossfor agent {} is {}".format(self.agent_id, actor_loss))
        if self.agent_id == 0:
            print('agent:{} crituc loss: {}, actor_loss: {}'.format(self.agent_id,critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(),1)
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),1)
        self.critic_optim.step()
        #self.writer.writerow(self.actor_network.action_out_cont.weight.grad)
        if self.train_step % 4==0:
            self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        #num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/actor_params.pkl')
        torch.save(self.critic_network.state_dict(), model_path + '/critic_params.pkl')


