from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple,  ActionSpec, BehaviorSpec, DecisionStep
import numpy as np
class dodgeball_agents:
    def __init__(self,file_name):
        self.file_name = file_name
        self.worker_id = 5
        self.seed = 4
        self.side_channels = []
        self.env=None
        self.nbr_agent=3
        self.spec=None
        self.agent_ids=[]
        self.agent_obs_size = 512
        self.num_envs = 1
        self.num_time_stacks = 3 #as defined in the build
        
    ##return the environment from the file
    def set_env(self):
        self.env=UnityEnvironment(file_name=self.file_name,worker_id=self.worker_id, seed=self.worker_id, side_channels=self.side_channels)
        self.env.reset()
        self.spec=self.team_spec()
        decision_steps,terminal_steps = self.env.get_steps(self.get_teamName())
        self.nbr_agent=len(decision_steps)
        self.agent_ids=list(decision_steps)
        return self.env
    
    ##specify the behaviour name for the corresponding team,here in this game id is either 0 or 1
    def get_teamName(self,teamId=0):
        assert teamId in [0,1]
        return list(self.env.behavior_specs)[1]

    ## define the specification of the observation and actions of the environment
    def team_spec(self):
        return self.env.behavior_specs[self.get_teamName()]

    ## continous and descrete actions
    def action_size(self):
        return self.spec.action_spec
    
    ## observation size in [(3, 8), (738,), (252,), (36,), (378,), (20,)] format
    def obs_size(self):
        return [self.spec.observation_specs[i].shape for i in range(len(self.spec.observation_specs))]

    #close the environment
    def env_close(self):
        self.env.close()

    ## move the environment to the next step
    def set_step(self):
        self.env.step()

    ## set the action for each agent
    def set_action_for_agent(self,agentId,act_continuous,act_discrete):
        assert type(act_continuous) == np.ndarray and type(act_discrete) == np.ndarray
        assert act_continuous.shape[1] == self.spec.action_spec.continuous_size and act_continuous.shape[0] == 1 \
                and act_discrete.shape[1] == self.spec.action_spec.discrete_size and act_discrete.shape[0] == 1
        action_tuple = ActionTuple()
        action_tuple.add_continuous(act_continuous)
        action_tuple.add_discrete(act_discrete)
        self.env.set_action_for_agent(self.get_teamName(),agentId, action_tuple)

    ##set the action for all agents
    def set_action_for_team(self,act_continuous,act_discrete):
        assert type(act_continuous) == np.ndarray and type(act_discrete) == np.ndarray
        assert act_continuous.shape[1] == self.spec.action_spec.continuous_size and act_continuous.shape[0] == self.nbr_agent \
                and act_discrete.shape[1] == self.spec.action_spec.discrete_size and act_discrete.shape[0] == self.nbr_agent
        action_tuple = ActionTuple()
        action_tuple.add_continuous(act_continuous)
        action_tuple.add_discrete(act_discrete)
        self.env.set_actions(self.get_teamName(),action_tuple)
      
    
    
    ##returns decision step for single agent from decision steps, team_id (0 or 1) and agent_index(0 or 1 or 2)
    def get_agent_decision_step(self, decision_steps, team_id, agent_index):
        assert team_id in [0, 1]
        assert agent_index in range(self.nbr_agent)
        assert type(decision_steps) == DecisionSteps
        if self.num_envs > 1:
            agent_ids = ([37, 51, 68], [0, 19, 32]) #(blue, purple) 
        else:
            agent_ids = ([3, 4, 5], [0, 1, 2])
        return decision_steps[agent_ids[team_id][agent_index]]
        
    
    ##given a decision step corresponding to a particular agent, return the observation as a long 1 dimensional numpy array
    def get_agent_obs_with_n_stacks(self, decision_step, num_time_stacks=1):
        #TODO: ainitialize with a big enough result instead of repetitive concatenation
        assert num_time_stacks >= 1
        assert type(decision_step) == DecisionStep
        obs = decision_step.obs
        result = obs[0].reshape((-1,))
        for i in range(1, len(obs)-1):
            result = np.concatenate((result, obs[i][:int(obs[i].shape[0]/self.num_time_stacks*num_time_stacks)]))
        result = np.concatenate((result, obs[-1]))
        return result
    

    ##returns agent observation from team decision_steps
    def get_agent_obs_from_decision_steps(self, decision_steps, team_id, agent_index, num_time_stacks=1):
        decision_step = self.get_agent_decision_step(decision_steps, team_id, agent_index)
        return self.get_agent_obs_with_n_stacks(decision_step, num_time_stacks)
        
        
    ##returns concatenated team observation from team decision_steps
    def get_team_obs_from_decision_steps(self, decision_steps, team_id, num_time_stacks=1):
        team_obs = np.zeros(shape=(self.nbr_agent*self.agent_obs_size,))
        for idx in range(self.nbr_agent):
            team_obs[self.agent_obs_size*idx:self.agent_obs_size*(idx+1)] = self.get_agent_obs_from_decision_steps(decision_steps, team_id, idx, num_time_stacks)
        return team_obs
            
        
        

            
            
            
        


