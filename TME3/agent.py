import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import matplotlib

matplotlib.use("TkAgg")


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space, statedic):
        self.action_space = action_space
        self.statedic = statedic

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def reinitialise(self):
        pass


class Q_learning(object):
    def __init__(self, action_space, statedic):
        self.action_space = action_space
        self.statedic = statedic

        self.alpha = 0.5
        self.gamma = 0.95
        self.eps = 1
        self.last_obs = None
        self.last_a = None
        self.Q = dict()

    def act(self, obs, reward, done):
        obs = self.statedic[gridworld.GridworldEnv.state2str(obs)]

        if obs not in self.Q.keys():
            self.Q[obs] = [0, 0, 0, 0]

        self.update_Q(reward, obs)

        if done:
            self.eps *= 0.999
            self.alpha *= 0.9995
            return

        if np.random.random_sample() > self.eps:
            a = self.Q[obs].index(max(self.Q[obs]))
        else:
            a = np.random.choice(4)

        self.last_obs = obs
        self.last_a = a

        return a

    def update_Q(self, reward, obs):

        if self.last_obs == None or self.last_a == None:
            return

        a = self.Q[obs].index(max(self.Q[obs]))

        self.Q[self.last_obs][self.last_a] += self.alpha * (
                    reward + self.gamma * self.Q[obs][a] - self.Q[self.last_obs][self.last_a])


    def reinitialise(self):
        self.last_obs = None
        self.last_a = None


class Dyna_Q(object):
    def __init__(self, action_space):
        self.action_space = action_space

        self.alpha = 0.1
        self.alpha_r = 0.1
        self.alpha_p = 0.1

        self.gamma = 0.95
        self.eps = 0.2
        self.last_obs = None
        self.last_a = None
        self.Q = dict()
        self.R = dict()
        self.P = dict()

    def act(self, obs, reward, done):
        obs = np.array_str(obs)
        if obs not in self.Q.keys():
            self.Q[obs] = [0, 0, 0, 0]

        if (self.last_obs, self.last_a, obs) not in self.R.keys():
            self.R[(self.last_obs, self.last_a, obs)] = 0
            self.P[(self.last_obs, self.last_a, obs)] = 0

        if np.random.random_sample() > self.eps:
            a = self.Q[obs].index(max(self.Q[obs]))
        else:
            a = np.random.choice(4)

        self.update_Q(reward, obs)
        self.update_R(reward, obs)
        self.update_P(reward, obs)

        self.last_obs = obs
        self.last_a = a

        return a

    def update_Q(self, reward, obs):

        if self.last_obs == None or self.last_a == None:
            return

        a = self.Q[obs].index(max(self.Q[obs]))

        self.Q[self.last_obs][self.last_a] += self.alpha * (
                    reward + self.gamma * self.Q[obs][a] - self.Q[self.last_obs][self.last_a])

    def update_R(self, reward, obs):

        if self.last_obs == None or self.last_a == None:
            return

        self.R[(self.last_obs, self.last_a, obs)] += self.alpha_r * (reward - self.R[(self.last_obs, self.last_a, obs)])

    def update_P(self, reward, obs):

        if self.last_obs == None or self.last_a == None:
            return

        self.P[(self.last_obs, self.last_a, obs)] += self.alpha_p * (reward - self.P[(self.last_obs, self.last_a, obs)])

        for triple in self.P:
            if triple[1] == self.last_a and triple[0] == self.last_obs:
                if triple[2] == obs:
                    pass
                self.P[triple] += self.alpha_p * (-self.P[triple])

    def reinitialise(self):
        self.last_obs = None
        self.last_a = None


class Sarsa(object):
    def __init__(self, action_space):
        self.action_space = action_space

        self.alpha = 0.1
        self.gamma = 0.95
        self.eps = 0.2
        self.last_obs = None
        self.last_a = None
        self.Q = dict()

    def act(self, obs, reward, done):
        obs = np.array_str(obs)
        if obs not in self.Q.keys():
            self.Q[obs] = [0, 0, 0, 0]

        if np.random.random_sample() > self.eps:
            a = self.Q[obs].index(max(self.Q[obs]))
        else:
            a = np.random.choice(4)
        self.update_Q(reward, obs, a)

        self.last_obs = obs
        self.last_a = a

        return a

    def update_Q(self, reward, obs, a):

        if self.last_obs == None or self.last_a == None:
            return

        self.Q[self.last_obs][self.last_a] += self.alpha * (
                    reward + self.gamma * self.Q[obs][a] - self.Q[self.last_obs][self.last_a])

    def reinitialise(self):
        self.last_obs = None
        self.last_a = None


if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random

    agent = Dyna_Q(env.action_space)
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        print(i)
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                agent.reinitialise()
                break

    print("done")
    env.close()

    """
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = RandomAgent(env.action_space)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
    """
