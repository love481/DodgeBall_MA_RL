from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from maddpg.agents import dodgeball_agents
from collections import deque
from typing import List
import csv
scores_average_window = 100    
class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.network_bank = deque(maxlen=self.args.size_netbank)
        self.current_network_bank=[]
        self.initial_elo=1000
        self.policy_elos=[self.initial_elo]* (self.args.size_netbank + 1)
        self.opponent_networks = None
        self.snapshot_counter=0
        self.current_opponent=0
        self.agents = [[], []]
        self._init_agents()
        self.buffer = Buffer(args)
        self.learning_team=0
        self.avg_returns_test={'team_purple':[],'team_blue':[]}
        self.avg_returns_train={'team_purple':[],'team_blue':[]}
        self.avg_elo_train={'team_purple':[],'team_blue':[]}
        self.scores_deque = {'team_purple':deque(maxlen=2),'team_blue':deque(maxlen=2)}
        self.episode_scores = []
        self.elo_deque = {'team_purple':deque(maxlen=1),'team_blue':deque(maxlen=1)}
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.f = open("returns.txt",'a')
        self.writer = csv.writer(self.f)
        self.episode_length=2000

    def _init_agents(self):
        for team_id in range(2):
            for i in range(self.args.n_learning_agents):
                self.agents[team_id].append(Agent(team_id*self.args.n_learning_agents + i, self.args))

    def compute_elo_rating_changes(self,rating,result):
        opponent_rating=self.policy_elos[self.current_opponent]
        r1 = pow(10, rating / 400)
        r2 = pow(10, opponent_rating/ 400)
        summed = r1 + r2
        e1 = r1 / summed
        if result is not None:
            change= result-e1
        else:
            change=0
        self.policy_elos[self.current_opponent] -= change
        self.policy_elos[-1] += change


    def swap_opponent_team(self):  
        for team_id in range(2):
            if team_id == self.learning_team:
                continue
            elif np.random.uniform() < (1-self.args.p_select_latest):
                idx = np.random.randint(len(self.network_bank)-1)
                self.opponent_networks = self.network_bank[idx]
            else:
                self.opponent_networks =  self.current_network_bank
                idx=-1
            self.current_opponent = idx
            #TODO load state dict to opponent team
            for agent_id in range(self.args.n_learning_agents):
                self.agents[1- self.learning_team][agent_id].load_actor_params(self.opponent_networks[agent_id])

    def select_action_for_opponent(self,agent_id):
            a=np.random.uniform(-1,1, self.args.action_shape[agent_id])*0.2
            #a=np.zeros(self.args.action_shape[agent_id])
            a[:self.args.continuous_action_space]= self.args.high_action*a[:self.args.continuous_action_space]
            a[self.args.continuous_action_space:]=(np.abs(a[self.args.continuous_action_space:])>0.2)*1
            return a

    def run(self):
        n=self.args.n_agents
        for episode in tqdm(range(1,self.episode_limit+1)):
            s,r,gr,d = self.env.reset()
            for agent in self.agents[0]:
                agent.noise.reset()
            rewards = {'team_purple':0,'team_blue':0}
            time_step = 0
            while True:
                a = []
                a_opponent = []
                with torch.no_grad():
                    for team_id in range(2):
                        for agent_id, agent in enumerate(self.agents[team_id]):
                            a.append(agent.select_action(s[team_id*self.args.n_learning_agents +agent_id], self.noise, self.epsilon))
                s_next, r,gr, done= self.env.step(a)
                if any(done)==True:
                    r=list((np.array(r)+np.array(gr))-(time_step/self.args.time_steps))
                    self.episode_length=time_step
                else:
                    r=list((np.array(r)+np.array(gr)))      
                    self.episode_length=self.args.time_steps
                # for i in range(4):
                #     if  r[i]==0.02:
                #         r[i]=2
                   
                rewards['team_purple'] += np.sum(r[:2])
                rewards['team_blue'] += np.sum(r[2:])
                self.buffer.store_episode(s[:n], a[:n], r[:n], s_next[:n],done[:n])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size and time_step%self.args.learn_rate==0:
                    experiences = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents[self.learning_team]:
                        other_agents = self.agents[self.learning_team].copy()
                        other_agents.remove(agent)
                        agent.learn(experiences, other_agents)
                if(any(done)==True) or time_step==2000:
                    break
                time_step += 1
            if rewards['team_purple']>2:
                result = 1.0
                purple_win=1
                blue_win=0
            elif rewards['team_blue']>2:
                result = 0.0
                blue_win=1
                purple_win=0
            else:
                result = 0.5
                purple_win=0
                blue_win=0
            self.compute_elo_rating_changes(self.policy_elos[-1], result)
            self.current_network_bank=[self.agents[self.learning_team][idx].get_actor_params() for idx in range(self.args.n_learning_agents)]
            self.noise = max(0.01, self.noise-((self.args.noise_rate-0.01)/self.episode_limit))
            self.epsilon = max(0.01, self.epsilon - 0.0015)
            self.scores_deque['team_blue'].append(rewards['team_blue'])
            self.scores_deque['team_purple'].append(rewards['team_purple'])
            self.elo_deque['team_blue'].append(self.policy_elos[self.current_opponent])
            self.elo_deque['team_purple'].append(self.policy_elos[-1])
            self.episode_scores.append(max(rewards['team_blue'],rewards['team_purple']))
            average_score = np.mean(self.episode_scores[episode-min(episode,scores_average_window):episode+1])
            rb=float(np.mean(self.scores_deque['team_blue']))
            rp=float(np.mean(self.scores_deque['team_purple']))
            eb=float(np.mean(self.elo_deque['team_blue']))
            ep=float(np.mean(self.elo_deque['team_purple']))
            self.writer.writerow([rb, rp, eb,ep,blue_win,purple_win,self.episode_scores[episode-1],average_score,self.agents[0][0].noise.sigma,self.episode_length])
            # print('team blue avg Returns is', np.mean(self.scores_deque['team_blue']))
            # print('team purple avg Returns is',np.mean(self.scores_deque['team_purple']))  
            # self.avg_returns_train['team_blue'].append(np.mean(self.scores_deque['team_blue']))
            # self.avg_returns_train['team_purple'].append(np.mean(self.scores_deque['team_purple']))
            # self.avg_elo_train['team_blue'].append(np.mean(self.elo_deque['team_blue']))
            # self.avg_elo_train['team_purple'].append(np.mean(self.elo_deque['team_purple']))
            if episode>0 and episode%20==0:
                self.network_bank.append([self.agents[self.learning_team][idx].get_actor_params() for idx in range(self.args.n_learning_agents)])
                self.policy_elos[self.snapshot_counter] =  self.policy_elos[-1]
                self.snapshot_counter = (self.snapshot_counter + 1) % self.args.size_netbank
            if episode>0 and episode%50==0:
                self.swap_opponent_team()
            # if episode>0 and episode%3==0:
            #     self.learning_team=(not self.learning_team)
            for team_id in range(2):
                for agent_id in range(self.args.n_learning_agents):
                    self.agents[team_id][agent_id].policy.save_model()
            if episode>0 and episode%10==0:
                self.f.flush()
            
            #     self.plot_graph(self.avg_returns_train,method='train_returns')
            #     self.plot_graph(self.avg_elo_train,method='train_elo')

        #return self.avg_returns_train
    
    def plot_graph(self,avg_returns,method=None):
        plt.figure()
        plt.plot(range(len(avg_returns['team_blue'])),avg_returns['team_blue'])
        plt.plot(range(len(avg_returns['team_purple'])),avg_returns['team_purple'])
        plt.xlabel('episode')
        plt.ylabel('average_'+method)
        plt.legend(["team_blue","team_purple"])
        plt.savefig(self.save_path + '/' + method +'_plt.png' , format='png')
           

    def evaluate(self):
        n=self.args.n_agents
        for episode in tqdm(range(self.args.evaluate_episodes)):
                    # reset the environment
            rewards = {'team_purple':0,'team_blue':0}
            s,r,gr,d = self.env.reset()
            for time_step in range(self.args.evaluate_episode_len):
                a = []
                a_opponent = []
                with torch.no_grad():
                    for team_id in range(2):
                        for agent_id, agent in enumerate(self.agents[team_id]):
                            a.append(agent.select_action(s[team_id*self.args.n_learning_agents +agent_id],0,0))
                #             a_opponent.append(self.select_action_for_opponent(agent_id))
                # for j in range(int(self.args.n_agents)):
                #     a.append(a_opponent[j])
                s_next, r,gr, done= self.env.step(a)
                if any(done)==True:
                    r=list((np.array(r)+np.array(gr))-(time_step/self.args.evaluate_episode_len))
                else:
                    r=list((np.array(r)+np.array(gr)))
                print( np.sum(r[:2]))
                rewards['team_purple'] += np.sum(r[:2])
                rewards['team_blue'] += np.sum(r[2:])
                s = s_next
                if(any(done)==True):
                    break
            self.scores_deque['team_blue'].append(rewards['team_blue'])
            self.scores_deque['team_purple'].append(rewards['team_purple'])
            print('team blue Returns is',np.mean(self.scores_deque['team_blue']))
            print('team purple Returns is', np.mean(self.scores_deque['team_purple']))
        self.avg_returns_test['team_blue'].append(np.mean(self.scores_deque['team_blue']))
        self.avg_returns_test['team_purple'].append(np.mean(self.scores_deque['team_purple']))
