from runner import Runner
from common.arguments import get_args
from common.utils import make_env

if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    for i in range(10):
        returns = runner.evaluate()
        print('Average returns is', returns)
    #runner.run()
