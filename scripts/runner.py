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
        self.avg_returns_test={'team_purple':[],'team_blue':[]}
        self.avg_returns_train={'team_purple':[],'team_blue':[]}
        self.count_ep=0
        self.scores_deque = {'team_purple':deque(maxlen=5),'team_blue':deque(maxlen=5)}
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def select_action_for_opponent(self,agent_id):
            a=np.random.uniform(-1,1, self.args.action_shape[agent_id])*0.5
            #a=np.zeros(self.args.action_shape[agent_id])
            a[:self.args.continuous_action_space]= self.args.high_action*a[:self.args.continuous_action_space]
            a[self.args.continuous_action_space:]=(np.abs(a[self.args.continuous_action_space:])>0.2)*1
            return a

    def run(self):
        returns = {'team_purple':[],'team_blue':[]}
        n=self.args.n_agents
        for i in range(self.episode_limit+1):
            s,r,gr,d = self.env.reset()
            for agent in self.agents:
                agent.noise.reset()
            rewards = {'team_purple':0,'team_blue':0}
            for time_step in tqdm(range(self.args.time_steps)):
                a = []
                a_opponent = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        a.append(action)
                        a_opponent.append(self.select_action_for_opponent(agent_id))
                for j in range(self.args.n_agents):
                    a.append(a_opponent[j])
                s_next, r,gr, done= self.env.step(a)
                r=list((np.array(r)+np.array(gr)))
                rewards['team_purple'] += np.sum(r[:3])
                rewards['team_blue'] += np.sum(r[3:])
                # if (np.sum(r[:3])>0.0001):
                #     for _ in range(50):
                self.buffer.store_episode(s[:n], a[:n], r[:n], s_next[:n],done[:n])
                s = s_next[:n]
                if self.buffer.current_size >= self.args.batch_size and time_step%self.args.learn_rate==0:
                    experiences = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(experiences, other_agents)
                # if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                #     self.evaluate()
                #     self.plot_graph( self.avg_returns_test['team_blue'],list( self.avg_returns_test.keys())[0],method='test')
                #     self.plot_graph( self.avg_returns_test['team_purple'],list( self.avg_returns_test.keys())[1],method='test')
                if(any(done)==True):
                    break
            self.noise = max(0.05, self.noise - 0.0001)
            #self.epsilon = max(0.05, self.epsilon - 0.0001)
            #returns['team_blue'].append(rewards['team_blue'])
            #returns['team_purple'].append(rewards['team_purple'])
            self.scores_deque['team_blue'].append(rewards['team_blue'])
            self.scores_deque['team_purple'].append(rewards['team_purple'])
            print('team blue avg Returns is', np.mean(self.scores_deque['team_blue']))
            print('team purple avg Returns is',np.mean(self.scores_deque['team_purple']))  
            self.avg_returns_train['team_blue'].append(np.mean(self.scores_deque['team_blue']))
            self.avg_returns_train['team_purple'].append(np.mean(self.scores_deque['team_purple']))
            self.plot_graph(self.avg_returns_train,method='train')
            if(np.mean(self.scores_deque['team_purple'])>3 and i >100):
                break
        return self.avg_returns_train
    
    def plot_graph(self,avg_returns,method=None):
        plt.figure()
        # plt.plot(range(len(avg_returns['team_blue'])),avg_returns['team_blue'])
        plt.plot(range(len(avg_returns['team_purple'])),avg_returns['team_purple'])
        plt.xlabel('episode')
        plt.ylabel('average returns')
        #plt.legend(["blue_reward","purple_reward"])
        if method=='test':
            plt.savefig(self.save_path + '/' + 'test_plt.png' , format='png')
        else:
            plt.savefig(self.save_path + '/' + 'train_plt.png' , format='png')
           
        # np.save(self.save_path + '/' + name + ('%d' % self.count_ep) + '_team_blue_returns.pkl',self.avg_returns['team_blue'])
        # np.save(self.save_path + '/' + name + ('%d' % self.count_ep) + '_team_purple_returns.pkl',self.avg_returns['team_purple'])

    def evaluate(self):
        n=self.args.n_agents
        returns = {'team_purple':[],'team_blue':[]}
        for episode in range(self.args.evaluate_episodes):
                    # reset the environment
            rewards = {'team_purple':0,'team_blue':0}
            s,r,gr,d = self.env.reset()
            for time_step in range(self.args.evaluate_episode_len):
                a = []
                a_opponent = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id],0,0)
                        a.append(action)
                        a_opponent.append(self.select_action_for_opponent(agent_id))
                        if agent_id==0:
                            print(s[agent_id])
                for j in range(self.args.n_agents):
                    a.append(a_opponent[j])
                s_next, r,gr, done= self.env.step(a)
                r=list((np.array(r))+np.array(gr))
                rewards['team_purple'] += np.mean(r[:3])
                rewards['team_blue'] += np.mean(r[3:])
                s = s_next[:n]
                if(any(done)==True):
                    break
            returns['team_blue'].append(rewards['team_blue'])
            returns['team_purple'].append(rewards['team_purple'])
            print('team blue Returns is', rewards['team_blue'])
            print('team purple Returns is', rewards['team_purple'])
        self.avg_returns_test['team_blue'].append(np.mean(returns['team_blue']))
        self.avg_returns_test['team_purple'].append(np.mean(returns['team_purple']))
