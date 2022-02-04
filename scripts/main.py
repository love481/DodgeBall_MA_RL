from runner import Runner
from common.arguments import get_args
from common.utils import make_env

if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    # for i in range(100):
    #     runner.evaluate()
    #     runner.plot_graph(runner.avg_returns['team_blue'],list(runner.avg_returns.keys())[0])
    #     runner.plot_graph(runner.avg_returns['team_purple'],list(runner.avg_returns.keys())[1])
    runner.run()
