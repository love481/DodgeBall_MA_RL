import numpy as np
import inspect
import functools
from maddpg.agents import dodgeball_agents
import torch 
from enum import Enum
import random
import copy
def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper

class Initialization(Enum):
    Zero = 0
    XavierGlorotNormal = 1
    XavierGlorotUniform = 2
    KaimingHeNormal = 3  # also known as Variance scaling
    KaimingHeUniform = 4
    Normal = 5

_init_methods = {
    Initialization.Zero: torch.zero_,
    Initialization.XavierGlorotNormal: torch.nn.init.xavier_normal_,
    Initialization.XavierGlorotUniform: torch.nn.init.xavier_uniform_,
    Initialization.KaimingHeNormal: torch.nn.init.kaiming_normal_,
    Initialization.KaimingHeUniform: torch.nn.init.kaiming_uniform_,
    Initialization.Normal: torch.nn.init.normal_,
}

def linear_layer(
    input_size: int,
    output_size: int,
    kernel_init: Initialization = Initialization.XavierGlorotUniform,
    kernel_gain: float = 1.0,
    bias_init: Initialization = Initialization.Zero,
) -> torch.nn.Module:
    """
    Creates a torch.nn.Linear module and initializes its weights.
    :param input_size: The size of the input tensor
    :param output_size: The size of the output tensor
    :param kernel_init: The Initialization to use for the weights of the layer
    :param kernel_gain: The multiplier for the weights of the kernel. Note that in
    TensorFlow, the gain is square-rooted. Therefore calling  with scale 0.01 is equivalent to calling
        KaimingHeNormal with kernel_gain of 0.1
    :param bias_init: The Initialization to use for the weights of the bias layer
    """
    layer = torch.nn.Linear(input_size, output_size)
    if (
        kernel_init == Initialization.KaimingHeNormal
        or kernel_init == Initialization.KaimingHeUniform
    ):
        _init_methods[kernel_init](layer.weight.data, nonlinearity="linear")
    else:
        _init_methods[kernel_init](layer.weight.data)
    layer.weight.data *= kernel_gain
    _init_methods[bias_init](layer.bias.data)
    return layer

class GaussianNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, sigma=0.50, decay=0.9982):
        """Initialize parameters and noise process."""
        self.size = size
        self.sigma = sigma
        self.decay = decay
        self.seed = torch.manual_seed(seed)
        # self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.sigma = max(self.decay*self.sigma,0.02)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        return self.sigma * torch.randn(self.size)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.01, sigma=0.2356, sigma_min = 0.05, sigma_decay=0.9965):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


def make_env(args,name,time_scale,no_graphics):
    env = dodgeball_agents(name, time_scale, no_graphics)
    env.set_env()
    args.n_learning_agents = env.nbr_agent
    args.n_agents = env.nbr_agent
    args.obs_shape = [env.agent_obs_size for i in range( args.n_agents)] 
    args.action_shape =[env.spec.action_spec.discrete_size + env.spec.action_spec.continuous_size for i in range( args.n_agents)] 
    args.high_action = 1
    args.low_action = -1
    args.continuous_action_space = env.spec.action_spec.continuous_size
    args.discrete_action_space = env.spec.action_spec.discrete_size
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.seed=45
    return env, args
