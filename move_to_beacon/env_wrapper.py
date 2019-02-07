from pysc2.lib import actions, features

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL


class SC2EnvWrapper:
    """Env wrapper for DeepMind Mini Game MoveToBeacon."""

    def __init__(self, env, res):
        self._env = env
        self._res = res
        self.actions = [actions.FUNCTIONS.select_army, actions.FUNCTIONS.Move_screen]

        self.action_space = len(self.actions)
        self.observation_space = (self._res, self._res)

        self._observation = None

    def reset(self):
        self._observation = self._env.reset()[0].observation
        return self._get_observation(), self._get_action_mask()

    def _get_observation(self):
        player_relative = self._observation.feature_screen.player_relative
        marine = (player_relative == _PLAYER_SELF).astype(int)
        beacon = (player_relative == _PLAYER_NEUTRAL).astype(int)
        selected = self._observation.feature_screen.selected

        return [marine, beacon, selected]

    def _get_action_mask(self):
        available_actions = self._observation['available_actions']
        action_mask = [0] * self.action_space
        for index, action in enumerate(self.actions):
            if action.id.value in available_actions:
                action_mask[index] = 1
        return action_mask

    def step(self, action):
        timestep = self._env.step([action])[0]
        self._observation = timestep.observation
        return self._get_observation(), timestep.reward, timestep.last(), {'action_mask': self._get_action_mask()}
