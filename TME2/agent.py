import numpy as np


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, _):
        return self.action_space.sample()


def evalPolicy(Agent, eps=0.01):

    policy_1 = np.random.random(len(Agent.statedic))
    policy_2 = np.zeros(len(Agent.statedic))

    while True:

        for state in Agent.mdp.keys():

            action = Agent.policy[Agent.statedic[state]]

            for proba, etat, reward, _ in Agent.mdp[state][action]:
                policy_2[Agent.statedic[state]] += proba * (reward + Agent.gamma * policy_1[Agent.statedic[etat]])

        dif = policy_1 - policy_2

        if np.linalg.norm(dif) < eps:
            return policy_2

        policy_1 = policy_2
        policy_2 = np.zeros(len(Agent.statedic))


class PolicyIterationAgent:

    def __init__(self, action_space, statedic, mdp, discount=0.99):
        self.action_space = action_space
        self.statedic = statedic
        self.mdp = mdp
        self.gamma = discount
        self.policy = {val: 0 for val in self.statedic.values()}

    def learn_policy(self):

        policy_change = True
        while policy_change:
            policy_change = False

            policy = evalPolicy(self)
            for state in self.mdp.keys():  # Pour chaque etat

                action_state = [0 for _ in range(4)]

                for action in range(4):
                    for proba, etat, reward, _ in self.mdp[state][action]:
                        action_state[action] += proba * (reward + self.gamma * policy[self.statedic[etat]])
                best_action = action_state.index(max(action_state))

                if best_action != self.policy[self.statedic[state]]:  # On change l'action a effectuer si ce n'est pas la meuilleure
                    self.policy[self.statedic[state]] = best_action
                    policy_change = True

    def act(self, observation):
        return self.policy[self.statedic[observation]]


class ValueIterationAgent(PolicyIterationAgent):

    def learn_policy(self):

        policy = evalPolicy(self)

        policy_change = True
        while policy_change:
            policy_change = False

            for i in self.mdp.keys():
                l = np.zeros(4)
                for action in range(4):
                    for p, e, r, d in self.mdp[i][action]:
                        l[action] += p * (r + self.gamma * policy[self.statedic[e]])
                best = np.argmax(l)
                if best != self.policy[self.statedic[i]]:
                    self.policy[self.statedic[i]] = best
                    policy_change = True
