import sys
import argparse
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.features import Dimensions, AgentInterfaceFormat
from move_to_beacon.env_wrapper import SC2EnvWrapper
from move_to_beacon.policy import Policy
from move_to_beacon.model import Model
from move_to_beacon.runner import Runner
import absl.flags
absl.flags.FLAGS(sys.argv)


def main():
    interface_format = AgentInterfaceFormat(
        feature_dimensions=Dimensions(screen=(args.res, args.res), minimap=(1, 1)),
        use_feature_units=True,
    )

    env = SC2Env(
        map_name='MoveToBeacon',
        agent_interface_format=interface_format,
        step_mul=8,
        random_seed=args.seed
    )

    env = SC2EnvWrapper(env, args.res)

    model = Model(
        policy=Policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr=args.learning_rate,
        spatial_res=(5, 5),
        cliprange=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef
    )

    runner = Runner(
        env=env,
        model=model,
        nsteps=args.num_steps,
        gamma=args.gamma,
        lam=args.lam
    )

    for _ in range((int(args.num_timesteps) // args.num_steps) + 1):
        runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StarCraft II mini-games reinforcement learning agents.')
    parser.add_argument('--map', type=str, default='MoveToBeacon', help='StarCraft II mini-games map')
    parser.add_argument('--res', type=int, default=32, help='resolution of screen and minimap')
    parser.add_argument('--num-envs', type=int, default=2, help='number of environments')
    # batch_size = num_steps * num_envs
    parser.add_argument('--num-steps', type=int, default=128, help='number of steps per update')
    parser.add_argument('--num-cpus', type=int, default=1, help='number of cpus')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num-timesteps', type=int, default=1e6, help='number of timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='discounting factor')
    parser.add_argument('--lam', default=0.95, type=float, help='advantage estimation discounting factor')
    parser.add_argument('--ent-coef', default=0.001, type=float, help='policy entropy coefficient')
    parser.add_argument('--vf-coef', default=0.5, type=float, help='value function loss coefficient')
    parser.add_argument('--clip-range', default=0.2, type=float, help='clipping range')
    parser.add_argument('--vis', action='store_true', default=False, help='render')
    args = parser.parse_args()

    main()
