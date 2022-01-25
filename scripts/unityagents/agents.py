from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple,  ActionSpec,BehaviorSpec
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

    ##return the environment from the file
    def set_env(self):
        self.env=UnityEnvironment(file_name=self.file_name,worker_id=self.worker_id, seed=self.worker_id, side_channels=self.side_channels)
        self.env.reset()
        self.spec=self.team_spec()
        decision_steps,terminal_steps = self.env.get_steps(self.get_teamName())
        self.nbr_agent=len(decision_steps)
        self.agent_ids=list(decision_steps)
    
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
        


