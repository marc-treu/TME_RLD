import gym
import matplotlib.pyplot as plt
import torch
from agent import *


def main(episodes, Agent, env, random=False):
    running_reward = 10
    resultat = []

    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state

        for time in range(1000):

            action = Agent.act(state)

            if random:
                state, reward, done, _ = env.step(action)
            else:
                state, reward, done, _ = env.step(action.data.item())

            agent.policy.reward_episode.append(reward)
            if done:
                break
        
        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        Agent.update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

        resultat.append(time)

    return resultat


def format_resultat(liste, tranche=100):

    affiche_liste = [sum(liste[i * tranche - tranche:i * tranche]) / tranche for i in
                      range(1, int(len(liste) / tranche))]

    x = [i * tranche - tranche / 2 for i in range(1, int(len(liste) / tranche))]

    return affiche_liste, x


def affiche_resultat(liste, x, nom=['Default']):

    for l in liste:
        plt.plot(x, l)

    plt.legend(nom, loc='upper left')

    plt.show()


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)
    learning_rate = 0.01
    gamma = 0.99
    epochs = 650
    t = 10

    agent = Agent(env, gamma, learning_rate)
    res_agent = main(epochs, agent, env)

    agent_random = RandomAgent()
    res_random = main(epochs, agent_random, env, True)

    env.close()

    y_agent, x_agent = format_resultat(res_agent, tranche=t)
    y_random, x = format_resultat(res_random, tranche=t)

    affiche_resultat([y_agent, y_random], x, nom=['A2C', 'Random'])

