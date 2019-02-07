import numpy as np
import tensorflow as tf


def sample(probs):
    random_uniform = tf.random_uniform(tf.shape(probs))
    scaled_random_uniform = tf.log(random_uniform) / probs
    return tf.argmax(scaled_random_uniform, axis=1)


class Model(object):
    def __init__(self, policy, observation_space, action_space, lr, spatial_res, cliprange, ent_coef, vf_coef):
        self.policy = policy
        self.observation_space = observation_space
        self.action_space = action_space
        self.spatial_res = spatial_res

        self.model = self.policy(self.observation_space, self.action_space, self.spatial_res)

        self.action_mask = tf.placeholder(tf.float32, [None, action_space], name='action_mask')
        self.spatial_mask = tf.placeholder(tf.float32, [None], name='spatial_mask')

        self.action = tf.placeholder(tf.int32, [None], name='action')
        self.spatial_action = tf.placeholder(tf.int32, [None], name='spatial_action')
        self.returns = tf.placeholder(tf.float32, [None], name='returns')

        self.advantage = tf.placeholder(tf.float32, [None])
        self.old_probs = tf.placeholder(tf.float32, [None, self.action_space])
        self.old_probs_spatial = tf.placeholder(tf.float32, [None, self.spatial_res[0] * self.spatial_res[1]])
        self.old_value = tf.placeholder(tf.float32, [None])

        action_probs = self.compute_action_probs(self.model.probs)
        action_log_probs = self.compute_action_log_probs(action_probs)
        old_action_probs = self.compute_action_probs(self.old_probs)
        old_action_log_probs = self.compute_action_log_probs(old_action_probs)

        spatial_action_log_probs = self.compute_spatial_action_log_probs(self.model.probs_spatial)
        old_spatial_action_log_probs = self.compute_spatial_action_log_probs(self.old_probs_spatial)

        self.sampled_actions = sample(action_probs)
        self.sampled_spatial_actions = sample(self.model.probs_spatial)

        # policy loss
        old_log_probs = old_action_log_probs + old_spatial_action_log_probs
        log_probs = action_log_probs + spatial_action_log_probs
        ratio = tf.exp(old_log_probs - log_probs)
        ratio_clipped = tf.clip_by_value(ratio, 1.0 - cliprange, 1.0 + cliprange)
        policy_loss = -self.advantage * ratio
        policy_loss_clipped = -self.advantage * ratio_clipped
        pg_loss = tf.reduce_mean(tf.maximum(policy_loss, policy_loss_clipped))

        # value loss
        vf_loss = tf.reduce_mean(tf.squared_difference(self.model.values, self.returns))

        # entropy
        entropy = -tf.reduce_mean(
            tf.reduce_sum(self.model.probs * tf.log(self.model.probs + 1e-13), axis=1, keepdims=True)
        )

        self.loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def compute_action_probs(self, probs):
        action_probs = probs * self.action_mask
        action_probs /= tf.reduce_sum(action_probs, axis=1, keepdims=True)
        return action_probs

    def compute_action_log_probs(self, action_probs):
        action_log_probs = -tf.reduce_sum(
            tf.one_hot(self.action, self.action_space) *
            tf.log(action_probs + 1e-13), axis=1
        )
        return action_log_probs

    def compute_spatial_action_log_probs(self, spatial_probs):
        spatial_action_log_probs = -tf.reduce_sum(
            tf.one_hot(self.spatial_action, self.spatial_res[0] * self.spatial_res[1]) *
            tf.expand_dims(self.spatial_mask, axis=1) * tf.log(spatial_probs + 1e-13), axis=1
        )
        return spatial_action_log_probs

    def step(self, obs, available_actions):
        actions, spatial_actions, values, probs, probs_spatial = self.sess.run([
            self.sampled_actions, self.sampled_spatial_actions,
            self.model.values, self.model.probs, self.model.probs_spatial
        ], feed_dict={
            self.action_mask: available_actions,
            self.model.screen_self: np.asarray(obs[0]),
            self.model.screen_neutral: np.asarray(obs[1]),
            self.model.screen_selected: np.asarray(obs[2])
        })
        return actions, spatial_actions, values[0], probs[0], probs_spatial[0]

    def value(self, obs):
        return self.sess.run(self.model.values, feed_dict={
            self.model.screen_self: np.asarray([obs[0]]),
            self.model.screen_neutral: np.asarray([obs[1]]),
            self.model.screen_selected: np.asarray([obs[2]])
        })

    def train(self):
        pass
