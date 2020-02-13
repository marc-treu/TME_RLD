import matplotlib
import gym
import gridworld
from agent import *
from gym import wrappers, logger
import numpy as np
import copy

matplotlib.use("TkAgg")


def main():
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print("env.action_space", env.action_space)  # Quelles sont les actions possibles

    obs, r, done, info = env.step(0)
    print('obs =', obs)  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    print('gridworld.GridworldEnv.state2str(obs) =', gridworld.GridworldEnv.state2str(obs))
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic

    print('statedic =',statedic)
    print('statedic[gridworld.GridworldEnv.state2str(obs)] =', statedic[gridworld.GridworldEnv.state2str(obs)])

    # print("Nombre d'etats : ", len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    # state, transitions = list(mdp.items())[0]
    # print(state)  # un etat du mdp
    # print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}
    # # input('ok ?')
    # env.close()


if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic

    agent_random = RandomAgent(env.action_space)

    agent_policy = PolicyIterationAgent(env.action_space, statedic, mdp, discount=0.90)
    agent_policy.learn_policy()
    print('agent.policy =', agent_policy.policy)

    agent_value = ValueIterationAgent(env.action_space, statedic, mdp, discount=0.999)
    agent_value.learn_policy()
    agent_value.learn_policy()
    print('agent.policy =', agent_value.policy)

    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.01
    env.verbose = False

    resultat = []

    for agent in [agent_random, agent_policy, agent_value]:

        print(agent)

        reward_agent = []
        episode_agent = []

        for i in range(episode_count):
            obs = env.reset()

            # env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
            if env.verbose:
                env.render(FPS)
            j = 0
            rsum = 0
            while True:
                action = agent.act(gridworld.GridworldEnv.state2str(obs))
                obs, reward, done, _ = env.step(action)
                rsum += reward
                j += 1
                if env.verbose:
                    env.render(FPS)
                if done:
                    reward_agent.append(rsum)
                    episode_agent.append(j)
                    #print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                    break
        resultat.append([reward_agent, episode_agent])
        print("done")

    for reward, episode in resultat:
        print('reward moyen =', sum(reward)/len(reward), " | nombre d'action moyen =", sum(episode)/len(episode))

    env.close()
