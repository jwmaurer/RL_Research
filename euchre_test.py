import argparse
import pprint

import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import set_seed, get_device

def run(args):
    # Make environment
    env = rlcard.make(
        args.env,
        config={
            'seed': 42,
        }
    )

    # Seed numpy, torch, random
    set_seed(42)

    device = get_device()

    # Set agents
    agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
        )
    agents = [RandomAgent(env.num_actions) for _ in range(1, env.num_players)]
    agents.append(agent)
    env.set_agents(agents)

    # Generate data from the environment
    trajectories, player_wins = env.run(is_training=False)
    # Print out the trajectories
    print('\nTrajectories:')
    print(trajectories)
    print('\nSample raw observation:')
    pprint.pprint(trajectories[0][0]['raw_obs'])
    print('\nSample raw legal_actions:')
    pprint.pprint(trajectories[0][0]['raw_legal_actions'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
            'euchre'
        ],
    )

    args = parser.parse_args()

    run(args)