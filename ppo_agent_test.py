# Author: Jacob Maurer
# Date: 3/3/2026 14:17
# Purpose: Experiment with the RLcard library.
# Gathered from: https://github.com/datamllab/rlcard/blob/master/docs/toy-examples.md#deep-q-learning-on-blackjack 
import os
import argparse

from rlcard.agents.ppo_agent import PPOAgent
import torch
import copy

import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        agent = DQNAgent(
	        replay_memory_size=20000,
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[1024,512,750,500,128,64],
            device=device,
        )
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[256,128,256,64],
            q_mlp_layers=[256,128,256,64],
            device=device,
        )
    elif args.algorithm == 'ppo':
        from rlcard.agents import ppo_agent
        agent = ppo_agent.PPOAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[1024,512,750,500,128,64],
            device=device,
        )
    agents = [agent, EuchreRuleAgent(), EuchreRuleAgent(), EuchreRuleAgent()]
    # for _ in range(1, env.num_players):
        # agents.append(copy.deepcopy(agent))
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for index, trajs in enumerate(trajectories):
                if isinstance(agents[index], PPOAgent):
                    for ts in trajs:
                        agents[index].feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                for i in range(len(agents)):
                    if isinstance(agents[index], PPOAgent):
                        print("Logging...")
                        logger.log_performance(
                            episode,
                            tournament(
                                env,
                                args.num_eval_games,
                            )[i],
                            i
                        )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    # plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    for index in range(len(agents)):
        if isinstance(agents[index], PPOAgent):
            save_path = os.path.join(args.log_dir, f'model_{index}.pth')
            torch.save(agents[index], save_path)
            print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
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
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
            'ppo'
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_dqn_result/',
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)