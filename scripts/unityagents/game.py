from turtle import done
from agents import dodgeball_agents
import numpy as np
if __name__ == "__main__":
    env = dodgeball_agents("/home/brabeem/Documents/deepLearning/builds/singenv/sectf.x86_64")
    env.set_env()
    state = env.reset()
    for i in range(100):
        actions  = env.random_action()
        next_state,reward,done = env.step(actions)
        next_state = state
    env.close()