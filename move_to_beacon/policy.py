import tensorflow as tf


class Policy(object):
    def __init__(self, observation_space, action_space, spatial_res):
        height, width = observation_space
        self.screen_self = tf.placeholder(tf.float32, [None, height, width], name='screen_self')
        self.screen_neutral = tf.placeholder(tf.float32, [None, height, width], name='screen_neutral')
        self.screen_selected = tf.placeholder(tf.float32, [None, height, width], name='screen_selected')

        num_channels = 3
        inputs = tf.concat([self.screen_self, self.screen_neutral, self.screen_selected], axis=2)
        reshaped = tf.reshape(inputs, [tf.shape(inputs)[0], height * width * num_channels])
        hidden = tf.layers.dense(inputs=reshaped, units=256, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=hidden, units=action_space, activation=None)
        logits_spatial = tf.layers.dense(inputs=hidden, units=spatial_res[0]*spatial_res[1], activation=None)

        self.probs = tf.nn.softmax(logits)
        self.probs_spatial = tf.nn.softmax(logits_spatial)
        self.values = tf.layers.dense(inputs=hidden, units=1)[:, 0]
