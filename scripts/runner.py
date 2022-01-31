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
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
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
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s,r,d = self.env.reset(1)
            a = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    a.append(action)
                    actions.append(action)
            ##print(actions[2][self.env.spec.action_spec.continuous_size:])
            s_next, r, done= self.env.step(actions,1)
            if(any(done)==True):
                self.env.close()
            r = list(np.array(r)-0.01)
            self.buffer.store_episode(s, a, r, s_next)
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                experiences = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(experiences, other_agents)
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
        self.env.close()

    def evaluate(self):
        avg_returns=[]
        s,r,d = self.env.reset(1)
        for time_step in tqdm(range(self.episode_limit)):
            returns = []
            for episode in range(self.args.evaluate_episodes):
                # reset the environment
                rewards = 0
                for time_step in range(self.args.evaluate_episode_len):
                    actions = []
                    with torch.no_grad():
                        for agent_id, agent in enumerate(self.agents):
                            action = agent.select_action(s[agent_id], 0, 0)
                            actions.append(action)
                    s_next, r, done= self.env.step(actions,1)
                    if(any(done)==True):
                        self.env.close()
                    rewards += r[0]+r[1]+r[2]
                    rewards -=0.01
                    s = s_next
                returns.append(rewards)
                print('Returns is', rewards)
            avg_returns.append(sum(returns) / self.args.evaluate_episodes)
            plt.figure()
            plt.plot(range(len(avg_returns)), avg_returns)
            plt.xlabel('episode')
            plt.ylabel('average returns')
            plt.savefig(self.save_path + '/plt.png', format='png')
            np.save(self.save_path + '/returns.pkl', avg_returns)
        return sum(avg_returns)/self.episode_limit
