from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from maddpg.agents import dodgeball_agents
class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = [self._init_agents(0),self._init_agents(1)]
        self.buffer = [Buffer(args),Buffer(args)]
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self,team_id):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args,team_id)
            agents.append(agent)
        return agents

    def run(self):
        s={0:[],1:[]}
        for i in range(self.episode_limit):
            s[0],r,d = self.env.reset(0)
            s[1],r,d = self.env.reset(0)
            for time_step in tqdm(range(self.args.time_steps)):
                for j in range(2):
                    a = []
                    actions = []
                    with torch.no_grad():
                        for agent_id, agent in enumerate(self.agents[j]):
                            action = agent.select_action(s[j][agent_id], self.noise, self.epsilon)
                            a.append(action)
                            actions.append(action)
                    s_next, r, done= self.env.step(actions,j)
                    if(any(done)==True):
                        s[j],r,d = self.env.reset(j)
                    self.buffer[j].store_episode(s[j], a, r, s_next)
                    s[j] = s_next
                    if self.buffer[j].current_size >= self.args.batch_size:
                        experiences = self.buffer[j].sample(self.args.batch_size)
                        for agent in self.agents[j]:
                            other_agents = self.agents[j].copy()
                            other_agents.remove(agent)
                            agent.learn(experiences, other_agents,j)
                    self.noise = max(0.05, self.noise - 0.0005)
                    self.epsilon = max(0.05, self.epsilon - 0.0005)
        self.env.close()
    
    def plot_graph(self,avg_returns,name):
        plt.figure()
        plt.plot(range(len(avg_returns)),avg_returns)
        plt.xlabel('episode')
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/' + name + '.png' , format='png')
        np.save(self.save_path + '/returns.pkl', avg_returns)

    def evaluate(self):
        avg_returns={'team_blue':[],'team_purple':[]}
        s={0:[],1:[]}
        for i in range(self.args.evaluate_rate):
            returns = {'team_blue':[],'team_purple':[]}
            for episode in range(self.args.evaluate_episodes):
                    # reset the environment
                rewards = {'team_blue':0,'team_purple':0}
                s[0],r,d = self.env.reset(0)
                s[1],r,d = self.env.reset(1)
                for time_step in range(self.args.evaluate_episode_len):
                    for j in range(2):
                        actions = []
                        with torch.no_grad():
                            for agent_id, agent in enumerate(self.agents[j]):
                                action = agent.select_action(s[j][agent_id], 0, 0)
                                actions.append(action)
                        s_next, r, done= self.env.step(actions,j)
                        if j==0:
                            rewards['team_blue'] += r[0]+r[1]+r[2]
                        elif j==1:
                            rewards['team_purple'] += r[0]+r[1]+r[2]
                        if(any(done)==True):
                            s[j],r,d = self.env.reset(j)
                        s[j] = s_next
                returns['team_blue'].append(rewards['team_blue'])
                returns['team_purple'].append(rewards['team_purple'])
                print('team blue Returns is', rewards['team_blue'])
                print('team purple Returns is', rewards['team_purple'])
            avg_returns['team_blue'].append(np.mean(returns['team_blue']))
            avg_returns['team_purple'].append(np.mean(returns['team_purple']))
        self.plot_graph(avg_returns['team_blue'],list(avg_returns.keys())[0])
        self.plot_graph(avg_returns['team_purple'],list(avg_returns.keys())[1])
        self.env.close()
