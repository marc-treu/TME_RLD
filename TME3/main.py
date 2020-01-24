import matplotlib
import gym
import gridworld
from agent import *
from gym import wrappers, logger
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use("TkAgg")


def execute_agent(agent_object=RandomAgent, plan=0, episode_count=10_000, visu=False):
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random

    plan = "gridworldPlans/plan" + str(plan) + ".txt"
    env.setPlan(plan, {0: -0.001, 3: 1, 4: 5, 5: -1, 6: -1})
    env.reset()

    statedic, _ = env.getMDP()  # Ici on récupere le statedic pour simplement simplifier la lecture des états.

    agent = agent_object(env.action_space, statedic)

    reward = 0

    rsum_list = []
    asum_list = []

    FPS = 0.0001
    for i in tqdm(range(episode_count)):

        done = False
        obs = env.reset()
        env.verbose = (i % 5000 == 0 and i > 0)  # afficher 1 episode sur 1000

        rsum = 0  # Somme des rewards
        asum = 0  # Somme des actions

        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            asum += 1
            if env.verbose and visu and asum < 300:
                env.render(FPS)
            if done:
                agent.act(obs, reward, done)
                # print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(asum) + " actions")
                agent.reinitialise()
                done = False
                break

        rsum_list.append(rsum)
        asum_list.append(asum)

    print("score moyen =", sum(rsum_list) / episode_count)

    print("done")
    env.close()

    return rsum_list, asum_list


def main():
    epoque = 10000
    tranche = 100

    resultat = []
    for agent in [RandomAgent, Q_learning, Dyna_Q, Sarsa]:
        print(agent)
        res_rewards, res_actions = execute_agent(agent, plan=0, visu=False, episode_count=epoque)

        resultat.append(([sum(res_rewards[i * tranche - tranche:i * tranche]) / tranche for i in
                          range(1, int(epoque / tranche))],
                         [sum(res_actions[i * tranche - tranche:i * tranche]) / tranche for i in
                          range(1, int(epoque / tranche))]))

    moy = [i * tranche - tranche / 2 for i in range(1, int(epoque / tranche))]

    for resultat_agent in resultat:
        plt.plot(moy, resultat_agent[0])

    plt.legend(['Random', 'Q_learning', 'Dyna_Q', 'Sarsa'], loc='upper left')

    plt.show()


if __name__ == '__main__':
    #execute_agent(Sarsa, plan=2, visu=True, episode_count=10_000)
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
