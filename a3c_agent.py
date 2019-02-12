import numpy as np
import tensorflow as tf

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
            self.non_spatial_action, self.spatial_action, self.value = build_network()

    def step(self, obs):
        screen = np.array(obs.observation.screen, dtype=np.float32)
        minimap = np.array(obs.observation.minimap, dtype=np.float32)

