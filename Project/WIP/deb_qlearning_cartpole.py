# -*- coding: utf-8 -*-

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# >>> Transition(0,0,0,2)
# Transition(state=0, action=0, next_state=0, reward=2)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        '''
        self.memory.push(self.process_state(state),
                action,
                self.process_state(next_state) if not next_state is None else None,
                reward)
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNet(nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        n = 256
        self.fc1 = nn.Linear(4, n)
        self.fc2 = nn.Linear(n, 2)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=True)
        x = self.fc2(x)

        return x

class Agent:
    def __init__(self, env):

        self.env = env
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.num_episodes = 3000

        self.time_train=[]
        self.time_test= []

        self.n_actions = env.action_space.n
        self.episode_durations = []

        self.policy_net = QNet().to(device)

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self, state):
        cum=0;
        N=1
        #facciamo un ciclo per mediare sulla funzione stocastica Q che sono un tot di sample di dropout
        for i in range(0,N):
            cum=self.policy_net(state)+cum

        M=cum/N
        print(M)
        print(cum)
        return M.max(1)[1].view(1, 1)


    def process_state(self,state):
        return torch.from_numpy(state).unsqueeze(0).float().to(device)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        '''
        >>> a = Transition([1,2,3],0,0,2)
        >>> a
        Transition(state=[1, 2, 3], action=0, next_state=0, reward=2)
        >>> c=[a for k in range(5)]
        >>> Transition(*zip(*c))
        Transition(state=([1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]),
                   action=(0, 0, 0, 0, 0),
                   next_state=(0, 0, 0, 0, 0),
                   reward=(2, 2, 2, 2, 2))
        '''


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.policy_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        #expected_state_action_values = (state_action_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_policy_model(self):

        for i_episode in range(self.num_episodes):

            state = self.env.reset()
            #print('episode: {}'.format(i_episode))

            for t in count():
                self.env.render()
                action = self.select_action(self.process_state(state))
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)


                if done:
                    next_state = None

                self.memory.push(
                    self.process_state(state),
                    action,
                    self.process_state(next_state) if next_state is not None else None,
                    reward)

                state = next_state

                self.optimize_model()
                if (t>10000):
                    done=1
                if done:
                    self.episode_durations.append(t + 1)
                    #self.plot_durations()
                    break


            print('ended episode: {}, with time {}'.format(i_episode, t+1))
            self.time_train.append(t+1)

        self.save_model()
        print('Training completed')
        plt.figure(figsize=(10,5))
        plt.plot(range(len(self.time_train)), self.time_train, 'steelblue')
        plt.title('Training dropout')
        plt.savefig('Training_DO.pdf', bbox_inches = 'tight')
        plt.show()



    def save_model(self):
        torch.save(self.policy_net.state_dict(), "./qlearning_model_DO")

    def load_model(self):
        self.policy_net.load_state_dict(torch.load("./qlearning_model_DO", map_location=device))

    def test(self, num_episodes):
        print('Testing model:')

        for i_episode in range(num_episodes):


            state = self.env.reset()
            for t in count():
                self.env.render()

                action = self.select_action(self.process_state(state))
                state, reward, done, info = env.step(action.item())
                # print(action.item())
                if done : break

            print('ended episode: {}, with time {}'.format(i_episode, t+1))
            self.time_test.append(t+1)
        print('Testing completed')
        plt.figure(figsize=(10,5))
        plt.plot(range(len(self.time_test)), self.time_test, 'steelblue')
        plt.title('Test_DO')
        plt.savefig('Test_DO.pdf', bbox_inches = 'tight')
        plt.show()

if 0: # eps treshold plot
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200

    l=[]
    for steps_done in range(1000):
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        l.append(eps_threshold)
    plt.plot(l)
    plt.show()

if __name__ == '__main__':

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    env = gym.make('CartPole-v0').unwrapped

    env.reset()

    agent = Agent(env)

    # Training phase
    if 1:
        agent.train_policy_model()

    #Testing phase
    agent.load_model()
    agent.test(10)

    env.close()
