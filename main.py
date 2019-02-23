import sys
import threading
from absl import app, flags
import tensorflow as tf
from pysc2.env import sc2_env, available_actions_printer

from a3c_agent import A3CAgent
from rollout import rollout


DEVICE = ['/cpu:0']
LOCK = threading.Lock()
STEP = 0
FLAGS = flags.FLAGS
flags.DEFINE_string('map_name', 'MoveToBeacon', 'SC2LE mini-games map name')
flags.DEFINE_integer('num_envs', 1, 'Number of environments')
flags.DEFINE_integer('screen_resolution', 64, 'Resolution of the screen')
flags.DEFINE_integer('minimap_resolution', 64, 'Resolution of the minimap')
flags.DEFINE_integer('step_mul', 8, 'Number of game steps per agent step')
flags.DEFINE_bool('render', False, 'Whether to render with pygame')
flags.DEFINE_bool('train', True, 'Whether to train agents')
flags.DEFINE_integer('max_steps', int(1e5), 'Total steps for training')
flags.DEFINE_integer('episode_horizon', 60, 'Total steps for every agents')
flags.DEFINE_integer('save_interval', 10, 'Number of steps between saving events')
flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate')
flags.DEFINE_float('gamma', 0.99, 'Discounting factor')
flags.DEFINE_string('logdir', './logs', 'Logs output directory')
FLAGS(sys.argv)


def run_thread(agent, visualize):
    with sc2_env.SC2Env(
        map_name=FLAGS.map_name,
        agent_interface_format=[sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(
                screen=(FLAGS.screen_resolution, FLAGS.screen_resolution),
                minimap=(FLAGS.minimap_resolution, FLAGS.minimap_resolution)
            )
        )],
        step_mul=FLAGS.step_mul,
        visualize=visualize
    ) as env:

        replay_buffer = []
        for trajectory, done in rollout([agent], env, FLAGS.max_steps):
            if FLAGS.train:
                replay_buffer.append(trajectory)
                if done:
                    step = 0
                    with LOCK:
                        global STEP
                        STEP += 1
                        step = STEP
                    learning_rate = FLAGS.learning_rate * (1 - 0.9 * step / FLAGS.max_steps)
                    agent.update(replay_buffer, FLAGS.gamma, learning_rate, step)
                    replay_buffer = []

                    if step >= FLAGS.max_steps:
                        break


def main(argv):
    agents = []
    for i in range(FLAGS.num_envs):
        agent = A3CAgent(FLAGS.screen_resolution, FLAGS.minimap_resolution, FLAGS.train)
        agent.build_model(i > 0, DEVICE[0])
        agents.append(agent)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(FLAGS.logdir)
    for i in range(FLAGS.num_envs):
        agents[i].setup(sess, summary_writer)

    agent.initialize()

    threads = []
    for i in range(FLAGS.num_envs - 1):
        t = threading.Thread(target=run_thread, args=(agents[i], False))
        threads.append(t)
        t.daemon = True
        t.start()
    run_thread(agents[-1], FLAGS.render)

    for t in threads:
        t.join()


if __name__ == '__main__':
    app.run(main)
