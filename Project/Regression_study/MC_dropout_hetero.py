##############################  BAYESIAN DEEP LEARNING  ############################## 
############### Clarotto Lucia, Franchini Alessandro, Lamperti Letizia ############### 

###############             MC DROPOUT (HETEROSCEDASTIC)                 ###############
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_variable(var=(), cuda=False, volatile=False):
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
    log_coeff = -no_dim*torch.log(sigma) - 0.5*no_dim*np.log(2*np.pi)
    
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

# Neural Network 
class MC_Dropout_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_units, drop_prob):
        super(MC_Dropout_Model, self).__init__()
        
        # input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_prob = drop_prob
        
        # 3 hidden layers
        self.layer1 = nn.Linear(input_dim, num_units)
        self.layer2 = nn.Linear(num_units, num_units)
        self.layer3 = nn.Linear(num_units, 2*output_dim)

    
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
    def __init__(self, network, learn_rate, batch_size, weight_decay):
        
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        
        self.network = network
        self.network.to(device)
        
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=learn_rate, weight_decay=weight_decay)
        self.loss_func = log_gaussian_loss
    
    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=False)
        
        # reset gradient and total loss
        self.optimizer.zero_grad()
        
        output = self.network(x)
        loss = self.loss_func(output[:, :1], y, output[:, 1:].exp(), 1)
    
        loss.backward()
        self.optimizer.step()

        return loss
    
    def get_loss_and_rmse(self, x, y, num_samples):
        x, y = to_variable(var=(x, y), cuda=False)
        
        means, stds = [], []
        for i in range(num_samples):
            output = self.network(x)
            means.append(output[:, :1])
            stds.append(output[:, 1:].exp())
        
        means, stds = torch.cat(means, dim=1), torch.cat(stds, dim=1)
        mean = means.mean(dim=-1)[:, None]
        std = ((means.var(dim=-1) + stds.mean(dim=-1)**2)**0.5)[:, None]
        loss = self.loss_func(mean, y, std, 1)
        
        rmse = ((mean - y)**2).mean()**0.5

        return loss.detach().cpu(), rmse.detach().cpu()


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
y_train = torch.cat((my_function(x1).view(-1,1) + torch.randn_like(x1)*0.3, my_function(x2).view(-1,1)+ torch.randn_like(x2)*0.3, my_function(x3).view(-1,1)+ torch.randn_like(x3)*0.3,  my_function(x4).view(-1,1) + 2 + torch.randn_like(x4)*0.3),0)

# Plot of the function with training data
plt.figure(figsize=(10,5))
plt.plot(x_train.numpy(),y_train.cpu().numpy(),'.',markersize=10, label='x train')
plt.plot(x_val.numpy(),y_val.cpu().numpy(),'-',markersize=7, label='x test')
plt.legend(fontsize=20)
plt.title('Regression function')
plt.show()


x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)


# Parameters of simulation
num_epochs, batch_size = 7000, len(x_train)

# Definition of NN
net = MC_Dropout_Wrapper(network=MC_Dropout_Model(input_dim=1, output_dim=1, num_units=300, drop_prob=0.5),
                         learn_rate=1e-4, batch_size=batch_size, weight_decay=1e-2)


# Training (with fit)
for i in range(num_epochs):
    
    loss = net.fit(x_train, y_train)
    
    if i % 200 == 0:
        print('Epoch: %4d, Train loss = %7.3f' % (i, loss.cpu().data.numpy()/batch_size))

# Testing (with forward)
samples = []
noises = []  
for i in range(1000):
    preds = net.network.forward(x_val).cpu().data.numpy()
    samples.append(preds[:, 0])
    noises.append(np.exp(preds[:, 1]))
    
samples = np.array(samples)
noises = np.array(noises)
means = (samples.mean(axis = 0)).reshape(-1)

# Credibility intervals
CI=np.zeros((len(x_val),2))
for i in range(len(x_val)):
    CI[i,:]=pymc3.stats.hpd(samples[:,i])
# credible_interval=0.94, circular=False
    


######### PLOTS ######### 
    
plt.figure(figsize=(10,5))
plt.plot(x_val.cpu().numpy(),samples[:].squeeze().T, 'steelblue',alpha=0.01)
plt.plot(x_train[:(len(x_train)-n_outliers)],y_train[:(len(x_train)-n_outliers)],'r.',markersize=4,alpha=0.8, label='Train data')

### If there's outliers uncomment the following line
#plt.plot(x_train[(len(x_train)-n_outliers):],y_train[(len(x_train)-n_outliers):],'rx',markersize=6, label='Extra perturbated train data',alpha=0.8)

plt.plot(x_val.cpu().numpy(),samples[:].mean(0).squeeze().T, 'orange',alpha=1,linewidth=3,label='Posterior mean')
plt.plot(x_val.cpu().numpy(),CI[:,0].squeeze().T, 'darkorange', alpha=1, label='Credibility Interval')
plt.plot(x_val.cpu().numpy(),CI[:,1].squeeze().T, 'darkorange', alpha=1)
plt.plot(x_val.numpy(),y_val.cpu().numpy(),'lime',linewidth=2,alpha=0.6, label='Real function')
plt.ylim([-1.5,4])
plt.xlim([-6.5,6.5])
plt.legend(loc='upper right')

plt.title('MC Dropout Heteroscedastic')
### If there's outliers uncomment the following line and comment the previous
#plt.title('MC Dropout Heteroscedastic with Outliers')

### If you want to save the plots in the current path
#plt.savefig(my_path+'MC_dropout_hetero.pdf', bbox_inches = 'tight')
#plt.savefig(my_path+'MC_dropout_hetero_outliers.pdf', bbox_inches = 'tight')
plt.show()
