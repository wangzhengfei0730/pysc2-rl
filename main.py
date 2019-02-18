import sys
import threading
from absl import flags
import argparse
import tensorflow as tf
from pysc2.env import sc2_env, available_actions_printer

from a3c_agent import A3CAgent
from rollout import rollout

DEVICE = ['/cpu:0']
LOCK = threading.Lock()
STEP = 0
FLAGS = flags.FLAGS


def run_thread(agent, visualize):
    with sc2_env.SC2Env(
        map_name=args.map_name,
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            rgb_dimensions=sc2_env.Dimensions(
                screen=args.screen_resolution,
                minimap=args.minimap_resolution
            )
        ),
        step_mul=args.step_mul,
        visualize=visualize
    ) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)

        replay_buffer = []
        for trajectory, done in rollout([agent], env, args.max_steps):
            if args.train:
                replay_buffer.append(trajectory)
                if done:
                    step = 0
                    with LOCK:
                        global STEP
                        STEP += 1
                        step = STEP
                    learning_rate = args.learning_rate * (1 - 0.9 * step / args.max_steps)
                    # update agent's policy
                    replay_buffer = []

                    if step >= args.max_steps:
                        break


def main():
    agents = []
    for i in range(args.num_envs):
        agent = A3CAgent(args.screen_resolution, args.minimap_resolution, args.train)
        agent.build_model(i > 0, DEVICE[0])
        agents.append(agent)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(args.log_dir)
    for i in range(args.num_envs):
        agents[i].setup(sess, summary_writer)

    agent.initialize()

    threads = []
    for i in range(args.num_envs - 1):
        t = threading.Thread(target=run_thread, args=(agents[i], False))
        threads.append(t)
        t.daemon = True
        t.start()
    run_thread(agents[-1], args.render)

    for t in threads:
        t.join()


if __name__ == '__main__':
    FLAGS(sys.argv)

    parser = argparse.ArgumentParser(description='StarCraft II mini-games reinforcement learning a3c agents.')
    parser.add_argument('--map-name', type=str, default='MoveToBeacon', help='SC2LE mini-games map name')
    parser.add_argument('--num-envs', type=int, default=2, help='Number of environments')
    parser.add_argument('--screen-resolution', type=int, default=64, help='Resolution of the screen')
    parser.add_argument('--minimap-resolution', type=int, default=64, help='Resolution of the minimap')
    parser.add_argument('--step-mul', type=int, default=8, help='Number of game steps per agent step')
    parser.add_argument('--render', action='store_true', default=False, help='Whether to render with pygame')
    # parser.add_argument('--train', action='store_true', default=False, help='Whether to train agents')
    parser.add_argument('--train', action='store_true', default=True, help='Whether to train agents')
    parser.add_argument('--max-steps', type=int, default=int(1e5), help='Total steps for training')
    parser.add_argument('--episode-horizon', type=int, default=60, help='Total steps for every agents')
    parser.add_argument('--save-interval', type=int, default=1, help='Number of steps between saving events')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Logs output directory')
    args = parser.parse_args()
    
    main()
