from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple,  ActionSpec, BehaviorSpec, DecisionStep
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid


# Create the StringLogChannel class
class StringLogChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unityno_graphics=self.no_graphics
        """
        # We simply read a string from the message and print it.
        pass

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        pass
    
string_log = StringLogChannel()


class dodgeball_agents:
    def __init__(self,file_name,time_scale, no_graphics):
        # Create the side channel
        self.engine_config_channel = EngineConfigurationChannel()
        self.engine_config_channel.set_configuration_parameters(time_scale=time_scale)
        self.file_name = file_name
        self.worker_id = 5
        self.seed = 4
        self.side_channels = [string_log, self.engine_config_channel]
        self.env=None
        self.nbr_agent=2
        self.spec=None
        self.no_graphics = no_graphics
        #self.agent_obs_size = 512 #356##without stacking##
        self.agent_obs_size = 504
        self.num_envs = 1
        #self.num_time_stacks = 3 #as defined in the build
        self.num_time_stacks = 1
        self.decision_steps = {0:DecisionSteps, 1:DecisionSteps}
        self.terminal_steps =  {0:TerminalSteps, 1:TerminalSteps}
        #self.agent_ids=([0, 1, 2],[3, 4, 5])
        self.agent_ids=([0, 1],[2, 3])
        
    ##return the environment from the file
    def set_env(self):
        self.env=UnityEnvironment(file_name=self.file_name,worker_id=self.worker_id, seed=self.worker_id, side_channels=self.side_channels)
        self.env.reset()
        self.spec=self.team_spec() 
        self.decision_steps[0],self.terminal_steps[0] = self.env.get_steps(self.get_teamName(teamId = 0))
        self.decision_steps[1],self.terminal_steps[1]  = self.env.get_steps(self.get_teamName(teamId = 1))
        assert len(self.decision_steps[0]) == len(self.decision_steps[1])
        if self.num_envs > 1:
            self.agent_ids = ([0, 19, 32],[37, 51, 68]) #(purple, blue) 
        else:
            #self.agent_ids = ([0, 1, 2],[3, 4, 5])
            self.agent_ids=([0, 1],[2, 3])
    
    ##specify the behaviour name for the corresponding team,here in this game id is either 0 or 1
    def get_teamName(self,teamId=0):
        assert teamId in [0,1]
        return list(self.env.behavior_specs)[teamId]

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
    def close(self):
        self.env.close()


    ## set the action for each agent of respective team
    def set_action_for_agent(self,teamId,agentId,act_continuous,act_discrete):
        assert self.agent_ids[teamId][agentId] in self.terminal_steps[teamId].agent_id or \
            self.agent_ids[teamId][agentId] in self.decision_steps[teamId].agent_id
        assert type(act_continuous) == np.ndarray and type(act_discrete) == np.ndarray
        assert act_continuous.shape[1] == self.spec.action_spec.continuous_size and act_continuous.shape[0] == 1 \
                and act_discrete.shape[1] == self.spec.action_spec.discrete_size and act_discrete.shape[0] == 1
        action_tuple = ActionTuple()
        action_tuple.add_continuous(act_continuous)
        action_tuple.add_discrete(act_discrete)
        self.env.set_action_for_agent(self.get_teamName(teamId),self.agent_ids[teamId][agentId], action_tuple)

    ##set the action for all agents of the repective team
    def set_action_for_team(self,teamId,act_continuous,act_discrete):
        assert type(act_continuous) == np.ndarray and type(act_discrete) == np.ndarray
        assert act_continuous.shape[1] == self.spec.action_spec.continuous_size and act_continuous.shape[0] == self.nbr_agent \
                and act_discrete.shape[1] == self.spec.action_spec.discrete_size and act_discrete.shape[0] == self.nbr_agent
        action_tuple = ActionTuple()
        action_tuple.add_continuous(act_continuous)
        action_tuple.add_discrete(act_discrete)
        self.env.set_actions(self.get_teamName(teamId),action_tuple)
        
        ##given a decision step corresponding to a particular agent, return the observation as a long 1 dimensional numpy array
    def get_agent_obs_with_n_stacks(self, decision_step, num_time_stacks=1):
        #TODO: ainitialize with a big enough result instead of repetitive concatenation
        assert num_time_stacks >= 1
        obs = decision_step.obs
        # print(obs[0].shape) ## (246,)
        # print(obs[1].shape) ## (84,)
        # print(obs[2].shape) ## (2,8)
        # print(obs[3].shape) ## (12,)
        # print(obs[4].shape) ## (20,)
        # print(obs[5].shape) ## (126,)
        result = np.concatenate((obs[0],obs[1]))
        result = np.concatenate((result,obs[2].reshape((-1,))))
        for i in range(3, 6):
            result = np.concatenate((result, obs[i]))
        assert result.shape[0] == 504
        return result
    #given a decision step corresponding to a particular agent, return the observation as a long 1 dimensional numpy array
    # def get_agent_obs_with_n_stacks(self, decision_step, num_time_stacks=1):
    #     #TODO: ainitialize with a big enough result instead of repetitive concatenation
    #     assert num_time_stacks >= 1
    #     obs = decision_step.obs
    #     result = obs[0].reshape((-1,))
    #     for i in range(1, len(obs)-1):
    #         result = np.concatenate((result, obs[i][:int(obs[i].shape[0]/self.num_time_stacks*num_time_stacks)]))
    #     result = np.concatenate((result, obs[-1]))
    #     return result
    

    ##returns agent observation from team decision_steps
    def get_agent_obs_from_decision_steps(self, decision_steps, team_id, agent_index, num_time_stacks=1):
        decision_step = decision_steps[self.agent_ids[team_id][agent_index]]
        return self.get_agent_obs_with_n_stacks(decision_step, num_time_stacks)
        
        
    ##returns concatenated team observation from team decision_steps
    def get_team_obs_from_decision_steps(self, decision_steps, team_id, num_time_stacks=1):
        team_obs = np.zeros(shape=(self.nbr_agent*self.agent_obs_size,))
        for idx in range(self.nbr_agent):
            team_obs[self.agent_obs_size*idx:self.agent_obs_size*(idx+1)] = self.get_agent_obs_from_decision_steps(decision_steps, team_id, idx, num_time_stacks)
        return team_obs
            
    ##returns agent reward ##    
    def reward_terminal(self,team_id,agent_index):
        assert self.agent_ids[team_id][agent_index] in self.decision_steps[team_id].agent_id or \
             self.agent_ids[team_id][agent_index] in self.terminal_steps[team_id].agent_id
        if self.agent_ids[team_id][agent_index] in self.decision_steps[team_id].agent_id:
            reward = self.decision_steps[team_id].__getitem__(self.agent_ids[team_id][agent_index]).reward
            grp_reward=self.decision_steps[team_id].__getitem__(self.agent_ids[team_id][agent_index]).group_reward
            done = False
        elif self.agent_ids[team_id][agent_index] in self.terminal_steps[team_id].agent_id:
            reward = self.terminal_steps[team_id].__getitem__(self.agent_ids[team_id][agent_index]).reward
            grp_reward=self.terminal_steps[team_id].__getitem__(self.agent_ids[team_id][agent_index]).group_reward
            done = True
        return reward,grp_reward,done

    ##get all agent obs as a list where each element in the list corresponds to an agent's observation##
    def get_all_agent_obs_rewards_dones(self,teamID=None):
        obs = []
        rewards = []
        grp_rewards = []
        dones = []
        if teamID==None:
            for teamid in range(2):
                for agentIndex in range( self.nbr_agent):
                    reward,grp_reward,done=self.reward_terminal(teamid,agentIndex)
                    if done==False:
                        obs.append(self.get_agent_obs_from_decision_steps(self.decision_steps[teamid],teamid,agentIndex,1))
                    else:
                        obs.append(self.get_agent_obs_from_decision_steps(self.terminal_steps[teamid],teamid,agentIndex,1))
                    rewards.append(reward)
                    grp_rewards.append(grp_reward)
                    dones.append(done)
        else:
            for agentIndex in range( self.nbr_agent):
                reward,grp_reward,done=self.reward_terminal(teamID,agentIndex)
                if done==False:
                    obs.append(self.get_agent_obs_from_decision_steps(self.decision_steps[teamID],teamID,agentIndex,1))
                else:
                    obs.append(self.get_agent_obs_from_decision_steps(self.terminal_steps[teamID],teamID,agentIndex,1))
                    ##print("exited yahooo {}".format(self.agent_ids[teamID][agentIndex]))
                rewards.append(reward)
                grp_rewards.append(grp_reward)
                dones.append(done)
        return obs,rewards,grp_rewards,dones

    ##reset the environment like gym##
    def reset(self,teamID=None):
        self.env.reset()
        if teamID==None:
            self.decision_steps[0],self.terminal_steps[0] = self.env.get_steps(self.get_teamName(teamId = 0))
            self.decision_steps[1],self.terminal_steps[1] = self.env.get_steps(self.get_teamName(teamId = 1))
            return self.get_all_agent_obs_rewards_dones()
        else:
            self.decision_steps[teamID],self.terminal_steps[teamID] = self.env.get_steps(self.get_teamName(teamId = teamID))
            return self.get_all_agent_obs_rewards_dones(teamID)

    ##step equivalent of gym environment##
    ##expects actions to be the list of action tuples and actiontuple is a namedtuple of continuous and 
    # discrete actions##
    def step(self,actions,teamId=None):
        ##set action for all agents##
        if teamId==None:
            for teamid in range(2):
                for agentInd in range( self.nbr_agent):
                    self.set_action_for_agent(teamid,agentInd,actions[ self.nbr_agent*teamid+agentInd][:self.spec.action_spec.continuous_size].reshape((1,self.spec.action_spec.continuous_size)) \
                    ,actions[ self.nbr_agent*teamid+agentInd][self.spec.action_spec.continuous_size:].reshape((1,self.spec.action_spec.discrete_size)))
                
        else:
            for agentInd in range( self.nbr_agent):
                self.set_action_for_agent(teamId,agentInd,actions[agentInd][:self.spec.action_spec.continuous_size].reshape((1,self.spec.action_spec.continuous_size)) \
                    ,actions[agentInd][self.spec.action_spec.continuous_size:].reshape((1,self.spec.action_spec.discrete_size)))
            ##get next_states ,rewards and dones from updated decision and terminal steps##

        self.env.step()
        if teamId==None:
            self.decision_steps[0],self.terminal_steps[0] = self.env.get_steps(self.get_teamName(0))
            self.decision_steps[1],self.terminal_steps[1] = self.env.get_steps(self.get_teamName(1))
            next_states,rewards,grp_rewards,dones  = self.get_all_agent_obs_rewards_dones()

        else:
            self.decision_steps[teamId],self.terminal_steps[teamId] = self.env.get_steps(self.get_teamName(teamId))
            next_states,rewards,grp_rewards,dones  = self.get_all_agent_obs_rewards_dones(teamId)

        return next_states,rewards,grp_rewards,dones


    
        
    
    
            
            
        

