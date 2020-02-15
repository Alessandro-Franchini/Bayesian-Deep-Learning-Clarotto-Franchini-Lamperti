##############################  BAYESIAN DEEP LEARNING  ############################## 
############### Clarotto Lucia, Franchini Alessandro, Lamperti Letizia ############### 

############### Q-LEARNING CARTPOLE PROBLEM  ###############
###############      Dropout method          ###############
###############  Analysis on uncertainty     ###############
 


######### LIBRARIES #########

import os
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
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Working only on cpu (NO CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# IMPORTANT
# modify the path where to save the Gif of the simulation 
my_path = os.path.dirname(__file__)
path_img = my_path + '/img/'



######### CLASSES #########


######################################################################
# Replay Memory
# -------------
#
# We will use experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data later. 
# By sampling from it randomly, the transitions that build up a batch are decorrelated. 
# This greatly stabilizes and improves the DQN training procedure.


#'Transition' is a named tuple representing a single transition in our environment. 
# It essentially maps (state, action) pairs to their (next_state, reward) result.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# 'ReplayMemory' is a cyclic buffer of bounded size that holds the
# transitions observed recently. It also implements a '.sample()'
# method for selecting a random batch of transitions for training.
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


######################################################################
# DQN model 
# -------------
      
# Neural Network
class QNet(nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        n = 128
        
        # 3 hidden layers
        self.fc1 = nn.Linear(4, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 2)

    def forward(self, x):
        
        ## # activation function (relu) on LAYER 1 (from input)
        x = F.relu(self.fc1(x))
        
        # dropout
        x = F.dropout(x, p=0.3, training=True)
        
        # activation function (relu) on LAYER 2 (from 1 to 3)
        x = F.relu(self.fc2(x))
        
        # dropout
        x = F.dropout(x, p=0.3, training=True)
        
        ## LAYER 3 (to output)
        x = self.fc3(x)

        return x

# Agent : creation
class Agent:
    def __init__(self, env):
        
        
        # Parameters 
        self.env = env
        self.batch_size = 128
        self.gamma = 0.999
        
        # The target network has its weights kept frozen most of the time, 
        # but is updated with the policy network's weights every so often (every 'target_update' timesteps') 
        self.target_update = 10
        
        #Number of episodes for the training
        self.num_episodes = 500
        
        ### Parameters for theepsilon-greedy algorithm (if needed)
#        self.eps_start = 0.9
#        self.eps_end = 0.05
#        self.eps_decay = 200
        
        self.time_train=[]
        self.time_train_mean=[]
        self.time_test= []

        self.n_actions = env.action_space.n
        self.episode_durations = []

        self.policy_net = QNet().to(device)
        self.target_net = QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        
    # 'select_action' selects an action accordingly to the dropout (Thompson sampling) policy. 
    def select_action(self, state):
        
        # Number of samples of dropout
        # You can play on this number (but no more than 15). It's the same N of the one in 'optimize_model'
        N=10                
        
        # Vector of samples
        vec = torch.zeros(N,2,device=device)
        cum=0

        # Make a cycle through the 'policy_net' in order to find the mean of the stochastic function Q, which contains N samples from the dropout
        for i in range(0,N):
            vec[i]=self.policy_net(state)
            cum=self.policy_net(state)+cum
        
        # Mean of the samples
        M=cum/N
        
        # Variance of the samples
        var = vec.var(0)
        
        # The action selected after the N samples 
        act = M.max(1)[1].view(1, 1)
        
        # Return the action, the mean and the variance of the samples
        return (act, M ,var)

    # utility 
    def process_state(self,state):
        return torch.from_numpy(state).unsqueeze(0).float().to(device)
    
    # 'plot_durations' is a helper for plotting the durations of episodes
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
            
    # 'optimize_model' is the function that performs a single step of the optimization. 
    # It first samples a batch, concatenates all the tensors into a single one, computes Q(s_t, a_t) 
    # and V(s_{t+1}) = max_a Q(s_{t+1}, a) and combines them into the loss. 
    # We also use a target network to compute V(s_{t+1}) for added stability. 
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)


        next_state_values = torch.zeros(self.batch_size, device=device)
        cum_next_state_values = torch.zeros(self.batch_size,2, device=device);
        
        # Number of samples of dropout
        # You can play on this number (but no more than 15). It's the same N of the one in 'select_action'
        N=10                        

        # Make a cycle through the 'target_net' in order to find the mean of the stochastic function Q, which contains N samples from the dropout
        for i in range(0,N):
            cum_next_state_values[non_final_mask]=self.target_net(non_final_next_states)+cum_next_state_values[non_final_mask]
            
        next_state_v=cum_next_state_values[non_final_mask]/N

        next_state_values[non_final_mask] = next_state_v.max(1)[0].detach()


        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss( state_action_values, expected_state_action_values.unsqueeze(1) )

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    
    # 'train_policy_model' is the MAIN TRAINING LOOP
    # At the beginning we reset the environment and initialize the `state` Tensor. 
    # Then, we sample an action, execute it, observe the next screen and the 
    # reward and optimize our model once. When the episode ends (our model
    # fails), we restart the loop.
    def train_policy_model(self):

        Nmean=20
        
        # Utility to know which episode to save
        counter=0
        for i_episode in range(self.num_episodes):
            
            
            # Save mean times
            if i_episode % Nmean == 0:                             
                Cumulative_time = 0
                Cumulative_mean=0
            
            
            state = self.env.reset()
            #print('episode: {}'.format(i_episode))
            
            
            # Number of the first "good" simulation (up for 700 timesteps) to be saved  
            n_sim = 700
            
            # Initialize vector of states, mean and variances
            Mean=torch.zeros(1500,2, device=device)
            Var = torch.zeros(1500,2, device=device)
            States = np.zeros((1500,4))

            # Initialize the vector of frames (images)
            frames = []
            for t in count():
                
                # Show the image 
                self.env.render()
                
                # Forward pass through the dropout (N dropout passes) in order to find the action, the mean and the variance
                action , mean , var = self.select_action(self.process_state(state))
                
                # Computation of the next_state and the reward 
                next_state, reward, done, _ = self.env.step(action.item())

                reward = torch.tensor([reward], device=device)
                
                # If no episode has been saved
                if counter==0:
                    
                    # Update vectors of mean, variance and state
                    Mean[t] = mean
                    Var[t] = var
                    States[t] = state
                    
                    # Update of the frames with the current state
                    frames.append(Image.fromarray(env.render(mode='rgb_array')))          
                    
                if done:
                    next_state = None
                
                self.memory.push(
                    self.process_state(state),
                    action,
                    self.process_state(next_state) if next_state is not None else None,
                    reward)
                
                # 'next_state' becomes the new 'state'
                state = next_state
                
                # Optimize model 
                self.optimize_model()
                
                # Stop the current episode if the time is more than 1500 (it's good enough)
                if (t==1499):
                    done=1
                if done:
                    self.episode_durations.append(t + 1)
                    #self.plot_durations()
                    break
                
                # IMPORTANT
                # Save just one "good" episode
                if (t==(n_sim-1) and counter==0):
                    
                    # Counter becomes 1, so that no more episodes are saved
                    counter=1
                    
                    # Create the dataframe with 4 variables of State, 2 variables of Mean and 2 variables of Variance
                    SMean=Mean[0:n_sim].detach().numpy()
                    SVar=Var[0:n_sim].detach().numpy()
                    df= pd.DataFrame({'Pos': States[0:n_sim,0],'Vel': States[0:n_sim,1],'Angle': States[0:n_sim,2],'Vel_tip': States[0:n_sim,3], 'Mean_1': SMean[:,0], 'Mean_2': SMean[:,1], 'Variance_1': SVar[:,0], 'Variance_2': SVar[:,1]})
                    
                    # Save the dataframe as .csv in the current path
                    s = 'file_df' 
                    df.to_csv(s+'.csv', index=False)
            
                    # Create and save the gif in the current path
                    file='/openai_gym.gif'
                    path=my_path + file
                    with open(path, 'wb') as f:  # change the path if necessary
                        im = Image.new('RGB', frames[0].size)
                        im.save(f, save_all=True, append_images=frames)
            
            # Update the target net every 'target_update' timesteps
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save_model()
            
            # Save the model after the completion of the episode
            print('ended episode: {}, with time {}'.format(i_episode, t+1))
            self.time_train.append(t+1)
            Cumulative_time=Cumulative_time+t+1
            
            #Save mean times for plot of the convergence
            if i_episode %Nmean==0:                             
                Cumulative_mean=Cumulative_time
                self.time_train_mean.append(t+1)
            
            
        # Save the model after the training phase 
        self.save_model()
        print('Training completed')
        
        # Plot the durations of the training episodes
        plt.figure(figsize=(10,5))
        plt.plot(range(len(self.time_train_mean)), self.time_train_mean, 'steelblue')
        plt.title('Training dropout')
        
        # Save the plot in the path 'path_img'
        plt.savefig(path_img+'Training_DO.pdf', bbox_inches = 'tight')
        plt.show()


    # Save the model with the updated weights
    def save_model(self):
        torch.save(self.policy_net.state_dict(), "./qlearning_model_DO")

    # Load an existing model
    def load_model(self):
        self.policy_net.load_state_dict(torch.load("./qlearning_model_DO", map_location=device))
    
    # Test the model ('num_episodes')-times
    def test(self, num_episodes):
        print('Testing model:')

        for i_episode in range(num_episodes):

            # Reset to the initial state
            state = self.env.reset()
            for t in count():
                self.env.render()
                
                # Pass through the network
                action = self.select_action(self.process_state(state))
                state, reward, done, info = env.step(action.item())
                
                if done : break

            print('ended episode: {}, with time {}'.format(i_episode, t+1))
            self.time_test.append(t+1)
            
        print('Testing completed')
        
        # Plot the durations of the testing episodes
        plt.figure(figsize=(10,5))
        plt.plot(range(len(self.time_test)), self.time_test, 'steelblue')
        plt.title('Test_DO')
        
        # Save the plot in the path 'path_img'
        plt.savefig(path_img+'Test_DO.pdf', bbox_inches = 'tight')
        plt.show()


### Epsilon-Greedy algorithm (if needed)
        
#if 0: # eps treshold plot
#    eps_start = 0.9
#    eps_end = 0.05
#    eps_decay = 200
#
#    l=[]
#    for steps_done in range(1000):
#        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
#        l.append(eps_threshold)
#    plt.plot(l)
#    plt.show()



######### MAIN #########
        
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

### If interested in the testing phase
        
#    #Testing phase
#    agent.load_model()
#    agent.test(10)

    env.close()
