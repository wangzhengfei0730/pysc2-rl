import os
import sys
import importlib
import threading
from absl import flags
import argparse
from functools import partial
import tensorflow as tf
from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import stopwatch

LOCK = threading.Lock()
FLAGS = flags.FLAGS


def run_thread():
    with sc2_env.SC2Env(
        map_name=args.map_name
    ) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)



def make_sc2env(**kwargs):
    env = sc2_env.SC2Env(**kwargs)
    return env


def main():
    print('arguments:', args)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    env_args = dict(
        map_name=args.map,
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=args.res, minimap=args.res)
        ),
        visualize=args.vis
    )

    env_fns = [partial(make_sc2env, **env_args)] * args.num_envs

    


if __name__ == '__main__':

    FLAGS(sys.argv)

    parser = argparse.ArgumentParser(description='StarCraft II mini-games reinforcement learning agents.')
    parser.add_argument('--map', type=str, default='MoveToBeacon', help='StarCraft II mini-games map')
    parser.add_argument('--res', type=int, default=32, help='resolution of screen and minimap')
    parser.add_argument('--num-envs', type=int, default=2, help='number of environments')
    # batch_size = num_steps * num_envs
    parser.add_argument('--num-steps', type=int, default=128, help='number of steps per update')
    parser.add_argument('--num-cpus', type=int, default=1, help='number of cpus')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num-timesteps', type=int, default=1e6, help='number of timesteps')

    parser.add_argument('--vis', action='store_true', default=False, help='render')
    args = parser.parse_args()
    
    main()
