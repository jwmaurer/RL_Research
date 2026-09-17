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
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, 10000)
    for position, reward in enumerate(rewards):
        print(position, models[position], reward)

if __name__ == '__main__':
    model_teams = [
        ('random', 'random'),
        ('rule_agent', 'rule_agent'),
        ('./experiments/dqn_more_mem_random_agent/model_0.pth', './experiments/dqn_more_mem_random_agent/model_2.pth'),
        ('./experiments/dqn_more_mem_rules_agent/model_0.pth', './experiments/dqn_more_mem_rules_agent/model_2.pth'),
        ('./experiments/euchre_dqn_train_all_more_memory/model_0.pth', './experiments/euchre_dqn_train_all_more_memory/model_2.pth'),
        ('./experiments/euchre_dqn_train_all_more_memory/model_1.pth', './experiments/euchre_dqn_train_all_more_memory/model_3.pth'),
        ('./experiments/ppo_new_dqn_agents/model_1.pth', './experiments/ppo_new_dqn_agents/model_3.pth'),
        ('./experiments/ppo_new_random_agent/model_0.pth', './experiments/ppo_new_random_agent/model_2.pth'),
        ('./experiments/ppo_new_rules_agent/model_0.pth', './experiments/ppo_new_rules_agent/model_2.pth'),
        ('./experiments/ppo_new_ppo_new/model_0.pth', './experiments/ppo_new_ppo_new/model_2.pth'),
        ('./experiments/ppo_new_ppo_new/model_1.pth', './experiments/ppo_new_ppo_new/model_3.pth'),
        ('./experiments/ppo_new_dqn_agents/model_0.pth', './experiments/ppo_new_dqn_agents/model_2.pth')
    ]
    
    for team1 in model_teams:
        for team2 in model_teams:
            test_eval = [team1[0], team2[0], team1[1], team2[1]]
            evaluate(test_eval)
            print()