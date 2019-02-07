import numpy as np
from pysc2.lib import actions


class Runner:

    def __init__(self, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma
        self.lam = lam

        self.observation, self.action_mask = self.env.reset()
        self.done = False
        self.advantage = 0

    def _generate_action(self, action_index, spatial_index):
        action = self.env.actions[action_index]
        if action == actions.FUNCTIONS.select_army:
            return actions.FUNCTIONS.select_army('select'), False
        else:
            y, x = np.unravel_index(spatial_index, self.model.spatial_res)
            return actions.FunctionCall(action.id, [[0], [y, x]]), True

    def _compute_advantage(self, t, done, next_value, rewards, values):
        if done:
            return rewards[t] - values[t]
        else:
            delta = rewards[t] + self.gamma * next_value - values[t]
            return delta + self.gamma * self.lam * self.advantage

    def run(self):
        mb_observations, mb_rewards, mb_dones = [], [], []
        mb_actions, mb_action_mask = [], []
        mb_spatial_actions, mb_spatial_mask = [], []
        mb_values, mb_probs, mb_spatial_probs = [], [], []

        for i in range(self.nsteps):
            action_index, spatial_index, value, prob, spatial_prob = self.model.step(
                np.asarray([self.observation]).swapaxes(0, 1),
                [self.action_mask]
            )
            action, spatial_mask = self._generate_action(action_index[0], spatial_index[0])

            mb_observations.append(self.observation.copy())
            mb_actions.append(action_index[0])
            mb_spatial_actions.append(spatial_index[0])
            mb_action_mask.append(self.action_mask)
            mb_spatial_mask.append(spatial_mask)
            mb_values.append(value)
            mb_probs.append(prob)
            mb_spatial_probs.append(spatial_prob)
            mb_dones.append(self.done)

            self.observation[:], rewards, self.done, infos = self.env.step(action)
            self.action_mask = infos['action_mask']
            mb_rewards.append(rewards)

        mb_advantage = np.zeros_like(mb_rewards)
        last_value = self.model.value(self.observation)[0]

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                self.advantage = self._compute_advantage(t, self.done, last_value, mb_rewards, mb_values)
            else:
                self.advantage = self._compute_advantage(t, mb_dones[t + 1], mb_values[t + 1], mb_rewards, mb_values)
            mb_advantage[t] = self.advantage
        mb_observations = np.asarray(mb_observations).swapaxes(0, 1)

        return \
            np.asarray(mb_observations), \
            np.asarray(mb_actions), \
            np.asarray(mb_action_mask), \
            np.asarray(mb_spatial_actions), \
            np.asarray(mb_spatial_mask), \
            np.asarray(mb_advantage), \
            np.asarray(mb_values), \
            np.asarray(mb_probs), \
            np.asarray(mb_spatial_probs)
