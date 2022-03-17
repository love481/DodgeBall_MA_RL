from math import fabs
from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import pickle
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
def store_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
tz="tz/small_map_touch_zone.x86_64"
tf="tf/small_map_touch_flag.x86_64"
rf="rf/small_map_return_flag.x86_64"
rf1="rf1/small_map_return_flags.x86_64"
if __name__ == '__main__':
    # get the params
    args = get_args()
    args.evaluate = 0
    time_scale = 2 if args.evaluate == True else 20
    no_graphics = False if args.evaluate == True else True
    env, args = make_env(args,"/home/love/Documents/" +tf,time_scale, no_graphics)
    #env, args = make_env(args,"/home/love/Downloads/envs/env-1",time_scale, no_graphics)
    runner = Runner(args, env)
    evaluate=args.evaluate
    if evaluate:
        for _ in range(10):
            runner.evaluate()
            runner.plot_graph(runner.avg_returns_test,method='test')
    else:
        runner.run()
        # store_data(args.save_dir + '/' + args.scenario_name +'/train_purple.txt',train['team_purple'])
        # store_data(args.save_dir + '/' + args.scenario_name +'/train_blue.txt',train['team_blue'])
