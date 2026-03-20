#
# Author: https://pettingzoo.farama.org/content/basic_usage/
# Purpose: to modify and to see if petting zoo can run multiagent
# Verdict: It can
# Date: 3/20/2026
# Time: 3:10

from pettingzoo.butterfly import cooperative_pong_v5

env = cooperative_pong_v5.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()