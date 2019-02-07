import numpy
from pysc2.agents import base_agent
from pysc2.lib import actions


FUNCTIONS = actions.FUNCTIONS
_PLAYER_NEUTRAL = 3


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


class MoveToBeacon(base_agent.BaseAgent):
    def step(self, obs):
        super(MoveToBeacon, self).step(obs)
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            if not beacon:
                return FUNCTIONS.no_op()
            beacon_center = numpy.mean(beacon, axis=0).round()
            return FUNCTIONS.Move_screen("now", beacon_center)
        else:
            return FUNCTIONS.select_army("select")
