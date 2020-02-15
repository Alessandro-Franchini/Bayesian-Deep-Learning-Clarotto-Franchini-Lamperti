##############################  BAYESIAN DEEP LEARNING  ############################## 
############### Clarotto Lucia, Franchini Alessandro, Lamperti Letizia ############### 

###############             MC DROPOUT (HOMOSCEDASTIC)                 ###############
###############              Regression Neural Network                 ###############
###############    Plots of samples, mean and credibility intervals    ###############



######### LIBRARIES #########

import os
import pymc3
import GPy
import time
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.sgd import SGD
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

# Working only on cpu (NO CUDA)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path for plots
my_path = os.path.dirname(__file__) + '/img/'



######### CLASSES #########

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out

# Loss function (Log Gaussian)
def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma)

    return -(log_coeff + exponent).sum()

# KL divergence
def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)

    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()

    return (varpost_lik*(varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))

        return (exponent + log_coeff).sum()

# Neural Network Layer
class MC_Dropout_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        super(MC_Dropout_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
    
    # Forward pass through the network
    def forward(self, x):

        dropout_mask = torch.bernoulli((1 - self.dropout_prob)*torch.ones(self.weights.shape)).to(device)

        return torch.mm(x, self.weights*dropout_mask) + self.biases

# Neural Network
class MC_Dropout_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, init_log_noise):
        super(MC_Dropout_Model, self).__init__()
        
        # input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 3 hidden layers
        self.layer1 = nn.Linear(input_dim, no_units)
        self.layer2 = nn.Linear(no_units, no_units)
        self.layer3 = nn.Linear(no_units, output_dim)

        self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]))


    def forward(self, x):

        x = x.view(-1, self.input_dim)
        
        ## LAYER 1 (from input)
        x = self.layer1(x)
        
        # activation function (sin)
        x = torch.sin(x)
        
        # dropout
        x = F.dropout(x, p=0.3, training=True)
        
        ## LAYER 2 (from 1 to 3)
        x = self.layer2(x)
        
        # activation function (relu)
        x = torch.relu(x)
        
        # dropout
        x = F.dropout(x, p=0.3, training=True)
        
        ## LAYER 3 (to output)
        x = self.layer3(x)
        
        return x


class MC_Dropout_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size, no_batches, weight_decay, init_log_noise):

        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches

        self.network = MC_Dropout_Model(input_dim = input_dim, output_dim = output_dim,
                                        no_units = no_units, init_log_noise = init_log_noise)
        self.network.to(device)

        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=learn_rate, weight_decay=weight_decay)
        self.loss_func = log_gaussian_loss

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=False)

        # reset gradient and total loss
        self.optimizer.zero_grad()

        output = self.network(x)
        loss = self.loss_func(output, y, torch.exp(self.network.log_noise), 1)/len(x)

        loss.backward()
        self.optimizer.step()

        return loss



######### SIMULATIONS #########

# Number of training points
N_tr = [100,100,100]

# Number of outliers
n_outliers = 0

# Number of testing points
N_val = 500

# Regression function
def my_function(x):
    y = torch.sin(3*(x+2)-(x+2)**2)+torch.exp(-0.2*x) 
    return y

# Training points and testing points
x_val = torch.linspace(-6.5,6.5,N_val).view(-1,1)
y_val = my_function(x_val).view(-1,1)
x1=torch.linspace(-4.5,-1.5,N_tr[0]).view(-1,1)
x2=torch.linspace(-1.5,1.5,N_tr[1]).view(-1,1)
x3=torch.linspace(1.5,4.5,N_tr[2]).view(-1,1)

# Position of the outliers
x4=torch.linspace(0.5,2,n_outliers).view(-1,1)

# Concatenating training data x
x_train = torch.cat( ( x1, x2, x3, x4 ) ,0)

# Adding some noise to the training data y (0.3*randn_like(xi))
y_train = torch.cat((my_function(x1).view(-1,1) + torch.randn_like(x1)*0.3,
                     my_function(x2).view(-1,1)+ torch.randn_like(x2)*0.3, my_function(x3).view(-1,1)+ torch.randn_like(x3)*0.3, my_function(x4).view(-1,1) + 2 + torch.randn_like(x4)*0.3),0)

# Plot of the function with training data
plt.figure(figsize=(10,5))
plt.plot(x_train.numpy(),y_train.numpy(),'.',markersize=10, label='x train')
plt.plot(x_val.numpy(),y_val.numpy(),'-',markersize=7, label='x test')
plt.legend(fontsize=20)
plt.title('Regression function')
plt.show()


x_train = x_train.to(device)
y_train = y_train.to(device)

x_val = x_val.to(device)
y_val = y_val.to(device)


# Parameters of simulation
num_epochs, batch_size, nb_train = 7000, len(x_train), len(x_train)

# Definition of NN
net = MC_Dropout_Wrapper(input_dim = 1, output_dim=1, no_units=200, learn_rate=1e-2,
                         batch_size=batch_size, no_batches=1, init_log_noise=0, weight_decay=1e-2)

# Training (with fit)
for i in range(num_epochs):

    loss = net.fit(x_train, y_train)

    if i % 200 == 0:
        print('Epoch: %4d, Train loss = %7.3f, noise = %6.3f' % \
              (i, loss.cpu().data.numpy(), torch.exp(net.network.log_noise).cpu().data.numpy()))


# Testing (with forward)
samples = []

for i in range(1000):
    preds = net.network.forward(x_val).cpu().data.numpy()
    samples.append(preds)

samples = np.array(samples)
means = (samples.mean(axis = 0)).reshape(-1)

# Credibility intervals
CI=np.zeros((len(x_val),2))
for i in range(len(x_val)):
    CI[i,:]=pymc3.stats.hpd(samples[:,i])
# credible_interval=0.94, circular=False
    
    
    
######### PLOTS #########
    
plt.figure(figsize=(10,5))
plt.plot(x_val.cpu().numpy(),samples[:].squeeze().T, 'steelblue',alpha=0.01)
plt.plot(x_train[:(len(x_train)-n_outliers)],y_train[:(len(x_train)-n_outliers)],'r.',markersize=4, label='Train data',alpha=0.8)

### If there's outliers uncomment the following line
#plt.plot(x_train[(len(x_train)-n_outliers):],y_train[(len(x_train)-n_outliers):],'rx',markersize=6, label='Extra perturbated train data',alpha=0.8)

plt.plot(x_val.cpu().numpy(),samples[:].mean(0).squeeze().T, 'orange',alpha=1,linewidth=3,label='Posterior mean')
plt.plot(x_val.cpu().numpy(),CI[:,0].squeeze().T, 'darkorange', alpha=1, label='Credibility Interval')
plt.plot(x_val.cpu().numpy(),CI[:,1].squeeze().T, 'darkorange', alpha=1)
plt.plot(x_val.cpu().numpy(),y_val.cpu().numpy(),'lime',linewidth=2,alpha=0.6,label='Real function')
plt.ylim([-2,5])
plt.xlim([-6.5,6.5])
plt.legend(loc='upper right')

plt.title('MC Dropout Homoscedastic')
### If there's outliers uncomment the following line and comment the previous
#plt.title('MC Dropout Homoscedastic with Outliers')

### If you want to save the plots in the current path
#plt.savefig(my_path+'MC_dropout_homo.pdf', bbox_inches = 'tight')
#plt.savefig(my_path+'MC_dropout_homo_outliers.pdf', bbox_inches = 'tight')
plt.show()
