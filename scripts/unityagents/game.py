from agents import dodgeball_agents
import numpy as np
if '__main__' == __name__:
    dodgeBall = dodgeball_agents("/home/love/Documents/ctfvideo/ctfvideo.x86_64")
    dodgeBall.set_env()
    print(dodgeBall.action_size())
    print(dodgeBall.nbr_agent)
    c=np.array([[0,0,1],[0,0,1],[0,0,1]])
    d=np.array([[0,0],[0,0],[0,0]])
    for i in range(100):
        # dodgeBall.set_action_for_agent(dodgeBall.agent_ids[0],np.array([[1,0,0]]),np.array([[0,1]]))
        dodgeBall.set_action_for_team(c,d)
        dodgeBall.set_step()
    dodgeBall.env_close()