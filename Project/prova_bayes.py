#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:52:05 2019

@author: luciaclarotto
"""

import hamiltorch
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_prob_func(params):
    mean = torch.tensor([1.,2.,3.])
    stddev = torch.tensor([0.5,0.5,0.5])
    return torch.distributions.Normal(mean, stddev).log_prob(params).sum()

num_samples = 400
step_size = .3
num_steps_per_sample = 5

hamiltorch.set_random_seed(123)
params_init = torch.zeros(3)
#params_hmc = hamiltorch.sample(log_prob_func=log_prob_func, params_init=params_init,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample)


#sampler=hamiltorch.Sampler.RMHMC
#integrator=hamiltorch.Integrator.IMPLICIT
#
#hamiltorch.set_random_seed(123)
#params_init = torch.zeros(3)
#params_irmhmc = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init, num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, sampler=sampler, integrator=integrator)

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
net = Net()
print(list(net.parameters()))

#torchvision.datasets.MNIST('/Users/letizialamperti/Documents/POLIMI/PROGETTO BAYES/Untitled Folder', train=True, transform=None, target_transform=None, download=True)

data_dir='/Users/ale/Desktop/Bayesian Statistics'
#datasets = torchvision.datasets.MNIST(root=data_dir)
#print(datasets)

import torchvision.datasets as datasets
mnist_trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=None)
plt.imshow(mnist_trainset.train_data[0].reshape((28,28)))
plt.show()
D = 784
N_tr = 100
N_val = 1000


x_train = mnist_trainset.train_data[:N_tr].float()/255.
x_train = x_train[:,None]
y_train = mnist_trainset.train_labels[:N_tr].reshape((-1,1)).float()
x_val = mnist_trainset.train_data[N_tr:N_tr+N_val].float()/255.
x_val = x_val[:,None]
y_val = mnist_trainset.train_labels[N_tr:N_tr+N_val].reshape((-1,1)).float()

x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

tau_list = []
tau = 0.001#./100. # 1/50
for w in net.parameters():
#     print(w.nelement())
#     tau_list.append(tau/w.nelement())
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

hamiltorch.set_random_seed(123)
net = Net()
params_init = hamiltorch.util.flatten(net).to(device).clone()
print(params_init.shape)

step_size = 0.008#0.01# 0.003#0.002
num_samples = 2000#2000 # 3000
L = 2 #3
tau_out = 1.
normalizing_const = 1.
burn =0 #GPU: 3000



#params_hmc = hamiltorch.sample_model(net, x_train, y_train, params_init=params_init, num_samples=num_samples,
 #                              step_size=step_size, num_steps_per_sample=num_steps_per_sample, tau_out=tau_out)




params_hmc = hamiltorch.sample_model(net, x_train, y_train, params_init=params_init, model_loss='multi_class_log_softmax_output', num_samples=num_samples, burn = burn,
                               step_size=step_size, num_steps_per_sample=L,tau_out=tau_out, tau_list=tau_list, normalizing_const=normalizing_const)



#from mlxtend.data import loadlocal_mnist

#
#
#X, y = loadlocal_mnist(
#        images_path="/Users/luciaclarotto/Desktop/MNIST",
#        labels_path="/Users/luciaclarotto/Desktop/MNIST")
#
#print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
#print('\n1st row', X[0])
