import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("TkAgg")
import gym
from gym import wrappers, logger
import numpy as np

from torch import nn
import torch
from collections import deque
import random

class NN(nn.Module):

    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
    
    def forward(self, x):
        x=torch.tensor(x, dtype=torch.float)
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x



class Agent:

    def __init__(self, maxlen, batch_size, gamma, C=10, lr=0.01):

        self.last_obs = None
        self.last_a = None
        self.batch_size = batch_size
        self.gamma = gamma
        self.criterion = torch.nn.SmoothL1Loss()
        self.iter = 0
        self.C = C

        self.Q = NN(4,2)
        self.optim = torch.optim.Adam(self.Q.parameters(),lr=lr)
        self.target = NN(4,2)
        self.D = deque(maxlen=maxlen)

    def reinitialise(self):
        self.last_obs = None
        self.last_a = None

    def act(self, observation, reward, done):
        if done:
            pass
        # print("self.last_obs", self.last_obs)
        # print("self.last_a", self.last_a)

        if self.last_obs is None or self.last_a is None:
            temp = self.Q.forward(observation)
            value, a = torch.max(temp, 0)

            self.last_obs = observation
            self.last_a = a.item()

            return a.item()

        self.D.append((self.last_obs, self.last_a, reward, observation, done))

        els = np.random.choice(range(len(self.D)), self.batch_size)
        batch = [self.D for i in els]
        ls0 = list()
        la= list()
        ls1 = list()
        lr = list()
        ld = list()
        for x in batch:
            ls0i, lai, ls1i, lri, ldi = x[0]
            ls0.append(ls0i)
            la.append(lai)
            ls1.append(ls1i)
            lr.append(lri)
            ld.append(ldi)
        # ls0, la, ls1, lr, ld = map(list, zip(*batch[0]))
        # print(batch[0])
        # la = map(list, zip(*batch))


        ls0 = torch.tensor(ls0)
        la = torch.tensor(la)

        q = self.Q.forward(ls0)

        q = q[range(self.batch_size), la]

        y = torch.zeros(self.batch_size)
        for i, x in enumerate(batch):
            x=x[0]
            if x[4]:
                y[i] = x[2]
            else:
                y[i] = x[2] + self.gamma * max(self.target.forward(x[3]))


        loss = self.criterion(q, y)
        loss.backward()
        self.optim.step()

        if self.iter % self.C == 0:
            self.target.load_state_dict(self.Q.state_dict())
            
        self.iter += 1

        # faire epsilon greedy
        temp = self.Q.forward(observation)
        value, a = torch.max(temp, 0)
        

        self.last_obs = observation
        self.last_a = a.item()

        return a.item()





if __name__ == '__main__':
    
    env = gym.make('CartPole-v1')

    agent = Agent(1000, 30, 0.95)
    # Enregistrement de l'Agent

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 100
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0
    FPS = 0.0001
    list_rsum = list()
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
                list_rsum.append(rsum)
                agent.reinitialise()
                break
    plt.figure()
    plt.plot(list_rsum)
    plt.savefig('dqn.png')
    print("done")
    env.close()
