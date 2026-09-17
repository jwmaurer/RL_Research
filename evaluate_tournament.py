"""
Jacob Maurer
6/17/2026
Purpose: Reformat the tournament function found in utils to also provide standard deviation
"""

import os
import argparse

import rlcard
from rlcard.agents import (
    DQNAgent,
    RandomAgent,
)
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device, weights_only=False)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    elif model_path == 'rule_agent':
        from rlcard.agents.euchre_rule_agent import EuchreRuleAgent
        agent = EuchreRuleAgent()
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    return agent

def evaluate(models):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(42)

    # Make the environment with seed
    env = rlcard.make('euchre', config={'seed': 42})

    # Load models
    agents = []
    for position, model_path in enumerate(models):
        agents.append(load_model(model_path, env, device=device))
    env.set_agents(agents)

    # Evaluate
    rewards_mean, rewards_std = tournament(env, 10000)
    for position, reward in enumerate(rewards_mean):
        print(position, models[position], reward, rewards_std[position])

if __name__ == '__main__':
    evals = [
        #Test against random agents to ensure they are playing correctly
        ['./experiments/ppo_new_ppo_new/model_0.pth', 'random', './experiments/ppo_new_ppo_new/model_2.pth', 'random'],
        ['random', './experiments/ppo_new_ppo_new/model_1.pth', 'random', './experiments/ppo_new_ppo_new/model_3.pth'],
        ['./experiments/ppo_new_dqn_agents/model_0.pth', 'random', './experiments/ppo_new_dqn_agents/model_2.pth', 'random'],
        ['random', './experiments/ppo_new_dqn_agents/model_1.pth', 'random', './experiments/ppo_new_dqn_agents/model_3.pth'],
        
        #Test against the rule based agent
        ['./experiments/ppo_new_ppo_new/model_0.pth', 'rule_agent', './experiments/ppo_new_ppo_new/model_2.pth', 'rule_agent'],
        ['rule_agent', './experiments/ppo_new_ppo_new/model_1.pth', 'rule_agent', './experiments/ppo_new_ppo_new/model_3.pth'],
        ['./experiments/ppo_new_dqn_agents/model_0.pth', 'rule_agent', './experiments/ppo_new_dqn_agents/model_2.pth', 'rule_agent'],
        ['rule_agent', './experiments/ppo_new_dqn_agents/model_1.pth', 'rule_agent', './experiments/ppo_new_dqn_agents/model_3.pth'],
        
        #Test the ppo-dqn trained agents versus the homogenous trained agents
        ['./experiments/euchre_dqn_train_all_more_memory/model_0.pth', './experiments/ppo_new_dqn_agents/model_1.pth', './experiments/euchre_dqn_train_all_more_memory/model_2.pth', './experiments/ppo_new_dqn_agents/model_3.pth'],
        ['./experiments/euchre_dqn_train_all_more_memory/model_1.pth', './experiments/ppo_new_dqn_agents/model_1.pth', './experiments/euchre_dqn_train_all_more_memory/model_3.pth', './experiments/ppo_new_dqn_agents/model_3.pth'],
        ['./experiments/ppo_new_dqn_agents/model_0.pth', './experiments/ppo_new_ppo_new/model_1.pth', './experiments/ppo_new_dqn_agents/model_2.pth', './experiments/ppo_new_ppo_new/model_3.pth'],
        ['./experiments/ppo_new_dqn_agents/model_0.pth', './experiments/ppo_new_ppo_new/model_0.pth', './experiments/ppo_new_dqn_agents/model_2.pth', './experiments/ppo_new_ppo_new/model_2.pth'],
    ]
    for eval in evals:
        evaluate(eval)
        print()