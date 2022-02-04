from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from maddpg.agents import dodgeball_agents
from collections import deque
class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.avg_returns_test={'team_blue':[],'team_purple':[]}
        self.avg_returns_train={'team_blue':[],'team_purple':[]}
        self.count_ep=0
        self.scores_deque = {'team_blue':deque(maxlen=5),'team_purple':deque(maxlen=5)}
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = {'team_blue':[],'team_purple':[]}
        for i in range(self.episode_limit+1):
            s,r,gr,d = self.env.reset()
            # for agent in self.agents:
            #     agent.noise.reset()
            rewards = {'team_blue':0,'team_purple':0}
            for time_step in tqdm(range(self.args.time_steps)):
                a = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        a.append(action)
                s_next, r,gr, done= self.env.step(a)
                r=list((np.array(r))+np.array(gr))
                rewards['team_blue'] += r[0]+r[1]+r[2]
                rewards['team_purple'] += r[3]+r[4]+r[5]
                self.buffer.store_episode(s, a, r, s_next,done)
                s= s_next
                if self.buffer.current_size >= self.args.batch_size:
                    experiences = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(experiences, other_agents)
                # if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                #     self.evaluate()
                #     self.plot_graph( self.avg_returns_test['team_blue'],list( self.avg_returns_test.keys())[0],method='test')
                #     self.plot_graph( self.avg_returns_test['team_purple'],list( self.avg_returns_test.keys())[1],method='test')
                self.noise = max(0.05, self.noise - 0.000005)
                self.epsilon = max(0.05, self.epsilon - 0.000005)
                if(any(done)==True):
                    break
            returns['team_blue'].append(rewards['team_blue'])
            returns['team_purple'].append(rewards['team_purple'])
            self.scores_deque['team_blue'].append(rewards['team_blue'])
            self.scores_deque['team_purple'].append(rewards['team_purple'])
            print('team blue avg Returns is', np.mean(self.scores_deque['team_blue']))
            print('team purple avg Returns is',np.mean(self.scores_deque['team_purple']))  
            self.avg_returns_train['team_blue'].append(np.mean(self.scores_deque['team_blue']))
            self.avg_returns_train['team_purple'].append(np.mean(self.scores_deque['team_purple']))
            self.count_ep+=1
        self.plot_graph( self.avg_returns_train['team_blue'],list(self.avg_returns_train.keys())[0],method='train')
        self.plot_graph(self.avg_returns_train['team_purple'],list(self.avg_returns_train.keys())[1],method='train')
    
    def plot_graph(self,avg_returns,name,method=None):
        plt.figure()
        plt.plot(range(len(avg_returns)),avg_returns)
        plt.xlabel('episode')
        plt.ylabel('average returns')
        if method=='test':
            plt.savefig(self.save_path + '/' + name + ('%d' % self.count_ep) + '_test_plt.png' , format='png')
        else:
            plt.savefig(self.save_path + '/' + name + ('%d' % self.count_ep) + '_train_plt.png' , format='png')
           
        # np.save(self.save_path + '/' + name + ('%d' % self.count_ep) + '_team_blue_returns.pkl',self.avg_returns['team_blue'])
        # np.save(self.save_path + '/' + name + ('%d' % self.count_ep) + '_team_purple_returns.pkl',self.avg_returns['team_purple'])

    def evaluate(self):
        returns = {'team_blue':[],'team_purple':[]}
        for episode in range(self.args.evaluate_episodes):
                    # reset the environment
            rewards = {'team_blue':0,'team_purple':0}
            s,r,gr,d = self.env.reset()
            for time_step in range(self.args.evaluate_episode_len):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id],0,0)
                        actions.append(action)
                s_next, r,gr, done= self.env.step(actions)
                r=list((np.array(r))+np.array(gr))
                rewards['team_blue'] += r[0]+r[1]+r[2]
                rewards['team_purple'] += r[3]+r[4]+r[5]
                s = s_next
                if(any(done)==True):
                    break
            returns['team_blue'].append(rewards['team_blue'])
            returns['team_purple'].append(rewards['team_purple'])
            print('team blue Returns is', rewards['team_blue'])
            print('team purple Returns is', rewards['team_purple'])
        self.avg_returns_test['team_blue'].append(np.mean(returns['team_blue']))
        self.avg_returns_test['team_purple'].append(np.mean(returns['team_purple']))
