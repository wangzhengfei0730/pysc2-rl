import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions, features

from network import build_network
import utils as U


class A3CAgent(object):
    """Asynchronous Advantage Actor-Critic agents for mini-games"""

    def __init__(self, screen_dimensions, minimap_dimensions, training, name='A3CAgent'):
        self.name = name
        self.training = training
        self.summary = []
        self.screen_dimensions = screen_dimensions
        self.minimap_dimensions = minimap_dimensions
        self.structured_dimensions = len(actions.FUNCTIONS)

    def reset(self):
        self.epsilon = [0.05, 0.2]

    def build_model(self, reuse, device):
        with tf.variable_scope(self.name) and tf.device(device):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # placeholder for inputs of network
            self.screen_ph = tf.placeholder(
                tf.float32,
                [None, U.screen_channel(), self.screen_dimensions, self.screen_dimensions],
                name='screen'
            )
            self.minimap_ph = tf.placeholder(
                tf.float32,
                [None, U.minimap_channel(), self.minimap_dimensions, self.minimap_dimensions],
                name='minimap'
            )
            self.structured_ph = tf.placeholder(tf.float32, [None, self.structured_dimensions], name='structured')

            # build network
            network = build_network(self.structured_ph, self.screen_ph, self.minimap_ph, len(actions.FUNCTIONS))
            self.non_spatial_action, self.spatial_action, self.value = network

            # placeholder for targets and masks
            self.valid_non_spatial_action_ph = tf.placeholder(
                tf.float32,
                [None, len(actions.FUNCTIONS)],
                name='valid_non_spatial_action'
            )
            self.sample_non_spatial_action_ph = tf.placeholder(
                tf.float32,
                [None, len(actions.FUNCTIONS)],
                name='sample_non_spatial_action'
            )
            self.valid_spatial_action_ph = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
            self.sample_spatial_action_ph = tf.placeholder(
                tf.float32,
                [None, self.minimap_dimensions ** 2],
                name='sample_spatial_action'
            )
            self.target_value_ph = tf.placeholder(tf.float32, [None], name='target_value')

            # compute log probability
            valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action_ph, axis=1)
            valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
            non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.sample_non_spatial_action_ph, axis=1)
            non_spatial_action_prob /= valid_non_spatial_action_prob
            non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
            spatial_action_prob = tf.reduce_sum(self.spatial_action * self.sample_spatial_action_ph, axis=1)
            spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
            self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))
            self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))

            # compute loss
            action_log_prob = self.valid_spatial_action_ph * spatial_action_log_prob + non_spatial_action_log_prob
            advantage = tf.stop_gradient(self.target_value_ph - self.value)
            policy_loss = -tf.reduce_mean(action_log_prob * advantage)
            value_loss = -tf.reduce_mean(self.value * advantage)
            loss = policy_loss + value_loss
            self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
            self.summary.append(tf.summary.scalar('value_loss', value_loss))

            # optimizer
            self.learning_rate_ph = tf.placeholder(tf.float32, None, name='learning_rate')
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate_ph, decay=0.99, epsilon=1e-10)
            grads = optimizer.compute_gradients(loss)
            clipped_grads = []
            for grad, var in grads:
                self.summary.append(tf.summary.histogram(var.op.name, var))
                self.summary.append(tf.summary.histogram(var.op.name + '/grad', grad))
                grad = tf.clip_by_norm(grad, 10.0)
                clipped_grads.append([grad, var])
            self.train_op = optimizer.apply_gradients(clipped_grads)
            self.summary_op = tf.summary.merge(self.summary)

            self.saver = tf.train.Saver(max_to_keep=None)

    def setup(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def step(self, obs):
        screen = np.array(obs.observation.feature_screen, dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
        minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
        structured = np.zeros([1, self.structured_dimensions], dtype=np.float32)
        structured[0, obs.observation.available_actions] = 1

        feed_dict = {
            self.screen_ph: screen,
            self.minimap_ph: minimap,
            self.structured_ph: structured
        }
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed_dict
        )

        non_spatial_action, spatial_action = non_spatial_action.ravel(), spatial_action.ravel()
        available_actions = obs.observation.available_actions
        action_id = available_actions[np.argmax(non_spatial_action[available_actions])]
        spatial_target = np.argmax(spatial_action)
        spatial_target = [int(spatial_target // self.screen_dimensions), int(spatial_target % self.screen_dimensions)]

        # epsilon-greedy exploration
        if self.training and np.random.rand() < self.epsilon[0]:
            action_id = np.random.choice(available_actions)
        if self.training and np.random.rand() < self.epsilon[1]:
            delta_y, delta_x = np.random.randint(-4, 5), np.random.randint(-4, 5)
            spatial_target[0] = int(max(0, min(self.screen_dimensions - 1, spatial_target[0] + delta_y)))
            spatial_target[1] = int(max(0, min(self.screen_dimensions - 1, spatial_target[1] + delta_x)))

        action_args = []
        for arg in actions.FUNCTIONS[action_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                action_args.append([spatial_target[1], spatial_target[0]])
            else:
                action_args.append([0])
        return actions.FunctionCall(action_id, action_args)

    def update(self, replay_buffer, gamma, learning_rate, step):
        obs = replay_buffer[-1][-1]
        if obs.last():
            reward = 0
        else:
            screen = np.array(obs.observation.feature_screen, dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            structured = np.zeros([1, self.structured_dimensions], dtype=np.float32)
            structured[0, obs.observation.available_actions] = 1

            feed_dict = {
                self.screen_ph: screen,
                self.minimap_ph: minimap,
                self.structured_ph: structured
            }
            reward = self.sess.run(self.value, feed_dict=feed_dict)

        # compute targets and masks
        screens, minimaps, structureds = [], [], []
        target_value = np.zeros([len(replay_buffer)], dtype=np.float32)
        target_value[-1] = reward

        valid_non_spatial_action = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.float32)
        sample_non_spatial_action = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.float32)
        valid_spatial_action = np.zeros([len(replay_buffer)], dtype=np.float32)
        sample_spatial_action = np.zeros([len(replay_buffer), self.screen_dimensions ** 2], dtype=np.float32)

        replay_buffer.reverse()
        for i, [obs, action, next_obs] in enumerate(replay_buffer):
            screen = np.array(obs.observation.feature_screen, dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            structured = np.zeros([1, self.structured_dimensions], dtype=np.float32)
            structured[0, obs.observation.available_actions] = 1

            screens.append(screen)
            minimaps.append(minimap)
            structureds.append(structured)

            reward = obs.reward
            action_id = action.function
            action_args = action.arguments

            target_value[i] = reward + gamma * target_value[i - 1]

            available_actions = obs.observation.available_actions
            valid_non_spatial_action[i, available_actions] = 1
            sample_non_spatial_action[i, action_id] = 1

            args = actions.FUNCTIONS[action_id].args
            for arg, action_arg in zip(args, action_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    spatial_action = action_arg[1] * self.screen_dimensions + action_arg[0]
                    valid_spatial_action[i] = 1
                    sample_spatial_action[i, spatial_action] = 1

        screens = np.concatenate(screens, axis=0)
        minimaps = np.concatenate(minimaps, axis=0)
        structureds = np.concatenate(structureds, axis=0)

        feed_dict = {
            self.screen_ph: screens,
            self.minimap_ph: minimaps,
            self.structured_ph: structureds,
            self.target_value_ph: target_value,
            self.valid_non_spatial_action_ph: valid_non_spatial_action,
            self.sample_non_spatial_action_ph: sample_non_spatial_action,
            self.valid_spatial_action_ph: valid_spatial_action,
            self.sample_spatial_action_ph: sample_spatial_action,
            self.learning_rate_ph: learning_rate
        }
        _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, step)

    def save_model(self, logdir, step):
        self.saver.save(self.sess, os.path.join(logdir, 'model'), global_step=step)

    def load_model(self, step):
        pass
