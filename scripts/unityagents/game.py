from agents import dodgeball_agents
import numpy as np
if '__main__' == __name__:
    dodgeBall = dodgeball_agents("/home/love/Documents/ctfvideo/ctfvideo.x86_64")
    dodgeBall.set_env()
    print(dodgeBall.action_size())
    print(dodgeBall.nbr_agent)
    c=np.array([[0,0,1],[0,0,1],[0,0,1]])
    d=np.array([[0,0],[0,0],[0,0]])
    agent_id=[0,1,2] ##agent_id is the index of the agent in the team
    team_id=[0,1] ##team_id is the index of the team
    for i in range(100):
        dodgeBall.set_action_for_agent(team_id[1],agent_id[0],np.array([[0.1,0,0]]),np.array([[0,1]]))
        #dodgeBall.set_action_for_team(team_id[0],c,d)
        dodgeBall.set_step(team_id[0])
        print(dodgeBall.get_agent_obs_from_decision_steps(dodgeBall.decision_steps,team_id[0], agent_id[0]))
        #print(dodgeBall.reward_and_terminalstate_for_agent(team_id[1],agent_id[0]))
        dodgeBall.env.reset()
    dodgeBall.env_close()