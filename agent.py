import torch
import torch.nn as nn
import random
import math
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Each agent have a memory
class Agent:
    def __init__(self, memory_capacity) -> None:
        # Generate double ended queue
        self.memory = deque([],maxlen=memory_capacity)


    def add_memory(self, *args):
        # Add transition to memory
        self.memory.append(Transition(*args))

    def get_memories(self, batch_size):
        # Get a random sample from memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class RandomAgent(Agent):
    """An example Random Agent"""

    def __init__(self, action_num,memory_capacity =1000) -> None:
        super().__init__(memory_capacity)
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        return self.policy.sample().item()


class DQNAgent(Agent):
    """A simple DQN Agent"""
    def __init__(self, action_num, Eps_end, Eps_start, Eps_decay,memory_capacity=1000) -> None:
        super().__init__(memory_capacity)
        self.action_num = action_num
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )
        self.EPS_END = Eps_end
        self.EPS_START = Eps_start
        self.EPS_DECAY = Eps_decay
        
    
    def act(self, state,policy_net, idx_step=0, force_policy=False):
        # Compute if we should use a random action or the policy_net
        epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * idx_step / self.EPS_DECAY)
        if random.random() > epsilon or force_policy:
            #use the policy_net with state
            return policy_net(state).max(1)[1][0].item()
        else:
            return self.policy.sample().item()

