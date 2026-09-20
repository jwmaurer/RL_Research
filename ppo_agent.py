#
# Author: Jacob Maurer
# Date: 3/27/2026
# Time: 9:10 pm
# Purpose: Implment PPO for the rlcard environment. 
# Help from: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# And: https://medium.com/@brianpulfer/ppo-intuitive-guide-to-state-of-the-art-reinforcement-learning-410a41cb675b
# 
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
import copy
from torch.distributions.categorical import Categorical 
from torch.nn.functional import softmax


class PPOAgent(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=1000,
                 num_actions=2,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005,
                 device=None,
                 save_path=None,
                 save_every=float('inf'),):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            replay_memory_size (int): Size of the replay memory
            replay_memory_init_size (int): Number of random experiences to sample when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps
            discount_factor (float): Gamma discount factor
            epsilon_start (float): Chance to sample a random action when taking an action.
              Epsilon is decayed over time and this is the start value
            epsilon_end (float): The final minimum value of epsilon after decaying is done
            epsilon_decay_steps (int): Number of steps to decay epsilon over
            batch_size (int): Size of batches to sample from the replay memory
            evaluate_every (int): Evaluate every N steps
            num_actions (int): The number of the actions
            state_space (list): The space of the state vector
            train_every (int): Train the network every X steps.
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            device (torch.device): whether to use the cpu or gpu
            save_path (str): The path to save the model checkpoints
            save_every (int): Save the model every X training steps
        '''
        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every

        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        self.ppo_estimator = Estimator( num_actions, learning_rate, state_shape, mlp_layers, device)

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)
        
        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(state['legal_actions'].keys()), done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()

    def step(self, state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        action, action_probs = self.predict(state)
        return action[0]

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        action, action_probs = self.predict(state)
        best_action = action 
        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(action_probs[0][list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return best_action, info

    def predict(self, state):
        ''' Predict the masked Q-values

        Args:
            state (numpy.array): current state

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value
        '''
        false_probs = np.zeros((1, 54))
        false_probs[0] = 1
        if(len(state["legal_actions"]) == 0):
            return 0, false_probs
        action, action_probs, _ = self.ppo_estimator.predict_nograd(np.expand_dims(state['obs'], 0), np.expand_dims(state["legal_actions"], 0))

        return action, action_probs

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        chosen_sample = self.memory.sample()
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        loss = self.ppo_estimator.update(chosen_sample, epsilon)
        print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately, 
            # add another argument to the function call parameterized by self.train_t
            self.save_checkpoint(self.save_path)
            print("\nINFO - Saved model checkpoint.")


    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if(len(legal_actions) == 0):
            return
        self.memory.save(state, action, reward, next_state, legal_actions, done)
        with torch.no_grad():
            _, action_probs, value_probs = self.ppo_estimator.predict_nograd(np.asarray([state]), np.asarray([legal_actions]))
        self.memory.save_probs_values(action_probs, value_probs)

    def set_device(self, device):
        self.device = device
        self.ppo_estimator.device = device

    def checkpoint_attributes(self):
        '''
        Return the current checkpoint attributes (dict)
        Checkpoint attributes are used to save and restore the model in the middle of training
        Saves the model state dict, optimizer state dict, and all other instance variables
        '''
        
        return {
            'agent_type': 'DQNAgent',
            'q_estimator': self.ppo_estimator.checkpoint_attributes(),
            'memory': self.memory.checkpoint_attributes(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'epsilon_start': self.epsilons.min(),
            'epsilon_end': self.epsilons.max(),
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'discount_factor': self.discount_factor,
            'update_target_estimator_every': self.update_target_estimator_every,
            'batch_size': self.batch_size,
            'num_actions': self.num_actions,
            'train_every': self.train_every,
            'device': self.device
        }
        
    @classmethod
    def from_checkpoint(cls, checkpoint):
        '''
        Restore the model from a checkpoint
        
        Args:
            checkpoint (dict): the checkpoint attributes generated by checkpoint_attributes()
        '''
        
        print("\nINFO - Restoring model from checkpoint...")
        agent_instance = cls(
            replay_memory_size=checkpoint['memory']['memory_size'],
            update_target_estimator_every=checkpoint['update_target_estimator_every'],
            discount_factor=checkpoint['discount_factor'],
            epsilon_start=checkpoint['epsilon_start'],
            epsilon_end=checkpoint['epsilon_end'],
            epsilon_decay_steps=checkpoint['epsilon_decay_steps'],
            batch_size=checkpoint['batch_size'],
            num_actions=checkpoint['num_actions'], 
            device=checkpoint['device'], 
            state_shape=checkpoint['q_estimator']['state_shape'],
            mlp_layers=checkpoint['q_estimator']['mlp_layers'],
            train_every=checkpoint['train_every']
        )
        
        agent_instance.total_t = checkpoint['total_t']
        agent_instance.train_t = checkpoint['train_t']
        
        agent_instance.ppo_estimator = Estimator.from_checkpoint(checkpoint['q_estimator'])
        agent_instance.memory = Memory.from_checkpoint(checkpoint['memory'])
        
        
        return agent_instance
                     
    def save_checkpoint(self, path, filename='checkpoint_dqn.pt'):
        ''' Save the model checkpoint (all attributes)

        Args:
            path (str): the path to save the model
        '''
        torch.save(self.checkpoint_attributes(), path + '/' + filename)

class Estimator(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.Estimator that
    uses PyTorch instead of Tensorflow.  All methods input/output np.ndarray.

    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    '''

    def __init__(self, num_actions=2, learning_rate=0.00001, state_shape=None, mlp_layers=None, device=None, g=0.9):
        ''' Initilalize an Estimator object.

        Args:
            num_actions (int): the number output actions
            state_shape (list): the shape of the state space
            mlp_layers (list): size of outputs of mlp layers
            device (torch.device): whether to use cpu or gpu
        '''
        self.num_actions = num_actions
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device
        self.gamma= g

        # set up Q model and place it in eval mode
        self.ppo_net = EstimatorNetwork(num_actions, state_shape, mlp_layers).to(device)
        self.ppo_net.eval()

        # initialize the weights using Xavier init
        for p in self.ppo_net.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up optimizer
        self.optimizer = torch.optim.Adam(self.ppo_net.parameters(), lr=self.learning_rate)
        
        #Optimal values found here: https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring18/slides/cs885-lecture15b.pdf
        self.c1 = 1
        self.c2 = 1e-2 

    def predict_nograd(self, s, legal):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions in the Double-DQN algorithm.

        Args:
          s (np.ndarray): (batch, state_len)

        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            action, action_probs, value_probs = self.ppo_net(s, legal)
        return action.cpu().numpy(), action_probs.cpu().numpy(), value_probs.cpu().numpy()

    def compute_cumulative_rewards(self, sample):
        """Given a buffer with states, policy action logits, rewards and terminations,
        computes the cumulative rewards for each timestamp and substitutes them into the buffer."""
        sample = list(sample)
        curr_rew = 0.
        # Traversing the buffer on the reverse direction
        for i in range(len(sample) - 1, -1, -1):
            sample[i] = list(sample[i])
            r, t = sample[i][2], sample[i][4]
            if t:
                curr_rew = 0
            curr_rew = r + self.gamma * curr_rew
            sample[i][2] = curr_rew
        # Normalizing cumulative rewards
        # mean = np.mean([sample[i][2] for i in range(len(sample))])
        # std = np.std([sample[i][2] for i in range(len(sample))]) + 1e-6
        # for i in range(len(sample)):
        #     sample[i][2] = (sample[i][2] - mean) / std
        return sample
    
    def get_losses(self, batch, epsilon):
        """Returns the three loss terms for a given model and a given batch and additional parameters"""
        # Getting old data
        n = len(batch)
        states = torch.cat([torch.from_numpy(np.asarray([batch[i][0]])) for i in range(n)]).to(self.device)
        legal_actions = torch.cat([torch.from_numpy(np.asarray([np.isin(np.asarray(range(self.num_actions)), np.asarray([batch[i][5]]))])) for i in range(n)]).to(self.device)
        actions = torch.cat([torch.tensor([batch[i][1]]) for i in range(n)]).view(n, 1).to(self.device)
        logits = torch.cat([torch.from_numpy(batch[i][-2]) for i in range(n)]).to(self.device)
        values = torch.cat([torch.from_numpy(batch[i][-1]) for i in range(n)]).to(self.device)
        cumulative_rewards = torch.tensor([batch[i][2] for i in range(n)]).view(-1, 1).float().to(self.device)
        # Computing predictions with the new model
        _, new_logits, new_values = self.ppo_net(states, legal_actions)

        # print(states.shape)
        # print(legal_actions.shape)
        # print(actions.shape)
        # print(logits.shape)
        # print(values.shape)
        # print(cumulative_rewards.shape)

        # Loss on the state-action-function / actor (L_CLIP)
        advantages = cumulative_rewards - values
        margin = epsilon
        ratios = new_logits.gather(dim=1, index=actions) / logits.gather(dim=1, index=actions)
        
        l_clip = torch.mean(
            torch.min(
                torch.cat(
                    (ratios * advantages,
                    torch.clip(ratios, 1 - margin, 1 + margin) * advantages),
                    dim=1),
                dim=1
            ).values
        )

        # Loss on the value-function / critic (L_VF)
        l_vf = torch.mean((cumulative_rewards - new_values) ** 2)

        # Bonus for entropy of the actor
        entropy_bonus = torch.mean(torch.sum(-new_logits * (torch.log(new_logits + 1e-9)), dim=1))

        return l_clip, l_vf, entropy_bonus

    def update(self, sample, epsilon):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()
        
        self.ppo_net.train()

        computed_sample = self.compute_cumulative_rewards(sample)

        l_clip, l_vf, l_entropy = self.get_losses(computed_sample, epsilon)
        # update model
        loss = l_clip - self.c1 * l_vf + self.c2 * l_entropy
        loss.backward()

        # Optimizing
        self.optimizer.step()

        self.ppo_net.eval()

        return loss.item()
    
    def checkpoint_attributes(self):
        ''' Return the attributes needed to restore the model from a checkpoint
        '''
        return {
            'ppo_net': self.ppo_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'state_shape': self.state_shape,
            'mlp_layers': self.mlp_layers,
            'device': self.device
        }
        
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restore the model from a checkpoint
        '''
        estimator = cls(
            num_actions=checkpoint['num_actions'],
            learning_rate=checkpoint['learning_rate'],
            state_shape=checkpoint['state_shape'],
            mlp_layers=checkpoint['mlp_layers'],
            device=checkpoint['device']
        )
        
        estimator.ppo_net.load_state_dict(checkpoint['ppo_net'])
        estimator.optimizer.load_state_dict(checkpoint['optimizer'])
        return estimator

class EstimatorNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    ''' 
    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the PPO network    
        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(EstimatorNetwork, self).__init__()    
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers    
        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        fc = [nn.Flatten()]
        fc.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        fa = copy.deepcopy(fc)
        self.actor_layers = nn.Sequential(*fa)
        self.critic_layers = nn.Sequential(*fc)
        self.device = "cuda" #Change later, to match global device
        
    def forward(self, s, legal_s):
        ''' Predict action values   
        Args:
            s  (Tensor): (batch, state_shape)
        '''
        action = self.actor_layers(s.type(torch.float))
        if isinstance(legal_s[0], dict):
            mask = torch.from_numpy(np.isin(np.asarray(range(54)), np.array(list(legal_s[0].keys()))))
        elif legal_s.shape[0] == 1:
            mask = torch.from_numpy(np.isin(np.asarray(range(54)), np.array(legal_s[0])))
        else:
            mask= legal_s
        mask = mask.to(self.device)
        masked_action = action.masked_fill(~mask, -np.inf)
        masked_action = torch.softmax(masked_action, dim=1)
        value = self.critic_layers(s.type(torch.float))
        chosen_action = Categorical(masked_action).sample()
        return chosen_action, masked_action, value

class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.ppo_memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = (state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)
        
    def save_probs_values(self, act_probs, act_values):
        if len(self.ppo_memory) == self.memory_size:
            self.ppo_memory.pop(0)
        transition = (act_probs, act_values)
        self.ppo_memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        sample_idxs = random.randint(0, len(self.memory) - self.batch_size) if len(self.memory) > self.batch_size else 0
        result_sample = []
        for sample_id in range(sample_idxs, sample_idxs + self.batch_size if sample_idxs + self.batch_size < len(self.memory) else len(self.memory)):
            result_sample.append(self.memory[sample_id] + self.ppo_memory[sample_id])
        return result_sample

    def checkpoint_attributes(self):
        ''' Returns the attributes that need to be checkpointed
        '''
        
        return {
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'memory': self.memory
        }
            
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' 
        Restores the attributes from the checkpoint
        
        Args:
            checkpoint (dict): the checkpoint dictionary
            
        Returns:
            instance (Memory): the restored instance
        '''
        
        instance = cls(checkpoint['memory_size'], checkpoint['batch_size'])
        instance.memory = checkpoint['memory']
        return instance
