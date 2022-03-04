import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="dodgeball", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=2000, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000, help="number of time steps")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.5, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.5, help="noise rate for sampling from a standard normal distribution")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(4e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=1000, help="number of episodes to optimize at the same time")
    parser.add_argument("--learn_rate", type=int, default=29, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=20, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=5, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=2000, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=1, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")


    #self-play
    parser.add_argument("--size_netbank", type=int, default=30, help="number of past actor networks to save")
    parser.add_argument("--swap_team", type=int, default=11, help="after this many steps, change the policy of opponent team")
    parser.add_argument("--save_team", type=int, default=10, help="after this many steps, save the current policy to network bank")
    parser.add_argument("--p_select_latest", type=float, default=0.5, help="probability with which to select the latest stored actor as opponent")
    args = parser.parse_args()

    return args
