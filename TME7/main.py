import gym
import numpy as np
import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from agent2 import *

# Hyperparameters
lr_mu = 0.0005
lr_q = 0.001
gamma = 0.99
buffer_limit = 50000
tau = 0.005  # for target network soft update


def main(epoch, render):
    env = gym.make('Pendulum-v0')
    memory = ReplayBuffer(buffer_limit)

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20
    scores = []

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(epoch):
        s = env.reset()

        for t in range(300):  # maximum length of episode is 200 for Pendulum-v0
            a = mu(torch.from_numpy(s).float())
            a = a.item() + ou_noise()[0]
            s_prime, r, done, info = env.step([a])
            memory.put((s, a, r / 100.0, s_prime, done))
            score += r
            s = s_prime

            if render:
                env.render()

            if done:
                scores.append(score)
                score = 0.0
                break

        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(scores[-print_interval:]) / print_interval))

    env.close()

    return scores


def main_random(epoch):

    env = gym.make('Pendulum-v0')

    score = 0.0
    print_interval = 1000
    scores = []

    for n_epi in range(epoch):
        s = env.reset()

        for t in range(300):  # maximum length of episode is 200 for Pendulum-v0

            _, r, done, _ = env.step([random.uniform(-2, 2)])
            score += r

            if done:
                scores.append(score)
                score = 0.0
                break

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(scores[-print_interval:]) / print_interval))

    env.close()

    return scores


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


if __name__ == '__main__':

    print('On apprend DDPG')
    ddpg = main(10_000, False)
    print('On exécute un agent Random')
    ran = main_random(10_000)

    t = 100

    y_ddpg, x_ddpg = format_resultat(ddpg, tranche=t)
    y_random, x = format_resultat(ran, tranche=t)

    affiche_resultat([y_ddpg, y_random], x, nom=['DDPG', 'Random'])
