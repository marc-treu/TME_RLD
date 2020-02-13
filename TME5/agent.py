import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import random


class RandomAgent:

    def __init__(self):
        pass

    def act(self, state):
        return random.randint(0, 1)

    def update_policy(self):
        pass


class Policy(nn.Module):

    def __init__(self, env, gamma):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma

        self.initialise = False

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.5),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


class Agent:

    def __init__(self, env, gamma, learning_rate):
        self.policy = Policy(env, gamma)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def act(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = self.policy(Variable(state))
        c = Categorical(state)
        action = c.sample()

        if self.policy.initialise is False:
            self.policy.policy_history = (c.log_prob(action))
            self.policy.initialise = True
            return action

        if self.policy.policy_history.dim() == 0:
            self.policy.policy_history = torch.cat(
                [self.policy.policy_history.unsqueeze(0), c.log_prob(action).unsqueeze(0)])
        else:
            self.policy.policy_history = torch.cat([self.policy.policy_history, c.log_prob(action).unsqueeze(0)])
        return action

    def update_policy(self):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.policy.reward_episode[::-1]:
            R = r + self.policy.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        loss = (torch.sum(torch.mul(self.policy.policy_history, Variable(rewards)).mul(-1), -1))

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        self.policy.loss_history.append(loss.data.item())
        self.policy.reward_history.append(np.sum(self.policy.reward_episode))
        self.policy.policy_history = Variable(torch.Tensor())
        self.policy.reward_episode = []