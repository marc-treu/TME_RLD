import matplotlib
import gym
import gridworld
from agent import *
from gym import wrappers, logger
import numpy as np
import copy
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def execute_agent(agent_object=RandomAgent):
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random

    statedic, _ = env.getMDP()  # Ici on récupere le statedic pour simplement simplifier la lecture des états.

    agent = agent_object(env.action_space, statedic)
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 10_000
    reward = 0
    done = False
    rsum_list = []
    FPS = 0.0001
    for i in range(episode_count):
        if i == 1000:
            print('1000')
        print(i)
        obs = envm.reset()
        env.verbose = (i % 5000 == 0 and i > 0)  # afficher 1 episode sur 1000
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
                agent.act(obs, reward, done)
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                agent.reinitialise()
                done = False
                break

        rsum_list.append(rsum)

    print(len(rsum_list))
    print(sum(rsum_list) / episode_count)

    print("done")
    env.close()
    print(len(rsum_list[1 * 10 - 10:1 * 10]))

    x = [i for i in range(episode_count)]
    moy = [i * 50 - 25 for i in range(1, 200)]
    rsum_list_moy = [sum(rsum_list[i * 100 - 100:i * 100]) / 100 for i in range(1, 100)]
    return rsum_list_moy

def main():

    rRandom = execute_agent()
    rQ_learning = execute_agent(Q_learning)

    moy = [i * 100 - 50 for i in range(1, 100)]

    plt.plot(moy, rRandom)
    plt.plot(moy, rQ_learning)

    plt.show()


if __name__ == '__main__':
    main()

    """
    
        env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print("env.action_space", env.action_space)  # Quelles sont les actions possibles

    obs, r, done, info =env.step(2)
    print(info)  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    # env.render()  # permet de visualiser la grille du jeu
    # env.render(mode="human")  # visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic

    # print("Nombre d'etats : ", len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    # state, transitions = list(mdp.items())[0]
    # print(state)  # un etat du mdp
    # print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}
    # # input('ok ?')
    # env.close()
    
    
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
    FPS = 0.01
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