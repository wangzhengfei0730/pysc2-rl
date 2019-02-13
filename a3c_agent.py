import numpy as np
import tensorflow as tf
from pysc2.lib import actions

from network import build_network
import utils as U


class A3CAgent(object):

    def __init__(self, non_spatial_dimensions, screen_dimensions, minimap_dimensions):
        self.non_spatial_dimensions = non_spatial_dimensions
        self.screen_dimensions = screen_dimensions
        self.minimap_dimensions = minimap_dimensions

    def build_model(self, reuse, device):
        with tf.variable_scope() and tf.device(device):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # placeholder for inputs of network
            self.non_spatial_ph = tf.placeholder(tf.float32, [None, self.non_spatial_dimensions], name='non_spatial')
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

            # build network
            network = build_network(self.non_spatial_ph, self.screen_ph, self.minimap_ph, len(actions.FUNCTIONS))
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

            # compute loss
            action_log_prob = self.valid_spatial_action_ph * spatial_action_log_prob + non_spatial_action_log_prob
            advantage = tf.stop_gradient(self.target_value_ph - self.value)
            policy_loss = -tf.reduce_mean(action_log_prob * advantage)
            value_loss = -tf.reduce_mean(self.value * advantage)
            loss = policy_loss + value_loss

            # optimizer
            self.learning_rate_ph = tf.placeholder(tf.float32, None, name='learning_rate')
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate_ph, decay=0.99, epsilon=1e-10)
            grads = optimizer.compute_gradients(loss)
            clipped_grads = []
            for grad, var in grads:
                grad = tf.clip_by_value(grad, 10.0)
                clipped_grads.append([grad, var])
            self.train_op = optimizer.minimize(clipped_grads)

    def step(self, obs):
        screen = np.array(obs.observation.screen, dtype=np.float32)
        minimap = np.array(obs.observation.minimap, dtype=np.float32)

