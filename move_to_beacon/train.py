import sys
import absl.flags
from pysc2.env import sc2_env
from pysc2.lib.features import Dimensions, AgentInterfaceFormat
from move_to_beacon.env_wrapper import SC2EnvWrapper

absl.flags.FLAGS(sys.argv)

interface_format = AgentInterfaceFormat(
    feature_dimensions=Dimensions(screen=(32, 32), minimap=(1, 1)),
    use_feature_units=True,
)

env = sc2_env.SC2Env(
    map_name='MoveToBeacon',
    agent_interface_format=interface_format,
    step_mul=8
)

env = SC2EnvWrapper(env, (32, 32))
env.reset()

