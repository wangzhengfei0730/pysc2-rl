import os
import sys
import threading
from absl import flags
import argparse
from functools import partial
import tensorflow as tf
from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import stopwatch

from a3c_agent import A3CAgent

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

    agents = []
    for i in range(None):
        agent = A3CAgent(None)
        agent.build_model(i > 0, None)
        agents.append(agent)

    config = tf.ConfigProto(allow_soft_placement=True)

    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(None)
    for i in range(None):
        agents[i].setup(sess, summary_writer)

    agent.initialize()

    threads = []
    for i in range(None):
        t = threading.Thread(target=run_thread, args=())
        threads.append(t)
        t.daemon = True
        t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':

    FLAGS(sys.argv)

    parser = argparse.ArgumentParser(description='StarCraft II mini-games reinforcement learning a3c agents.')
    parser.add_argument('--render', action='store_true', default=False, help='Whether to render with pygame')

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
