###############################
####       MC DROPOUT      ####
###############################


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def forward(self, x):

        dropout_mask = torch.bernoulli((1 - self.dropout_prob)*torch.ones(self.weights.shape)).to(device)

        return torch.mm(x, self.weights*dropout_mask) + self.biases

# Neural Network

class MC_Dropout_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, init_log_noise):
        super(MC_Dropout_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # only 1 hidden Layer
        self.layer1 = nn.Linear(input_dim, no_units)
        self.layer2 = nn.Linear(no_units, output_dim)

        # activation to be used between hidden layers
        #self.activation = nn.ReLU(inplace = True)
        self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]))


    def forward(self, x):

        x = x.view(-1, self.input_dim)

        x = self.layer1(x)
        #x = self.activation(x) ReLU
        x = torch.tanh(x)

        x = F.dropout(x, p=0.2, training=True)

        x = self.layer2(x)

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


# np.random.seed(2)
# no_points = 400
# lengthscale = 1
# variance = 1.0 # known variance
# sig_noise = 0.3
# x = np.random.uniform(-3, 3, no_points)[:, None]
# x.sort(axis = 0)

# Test regression

# N_tr = [50,3,50]
# N_val = 400
#
#
# x_val = torch.linspace(-10,10,N_val).view(-1,1)
# #y_val = (x_val**2).view(-1,1)
# y_val = torch.cos(x_val).view(-1,1)
# x1=torch.linspace(-6.28,-0.5,N_tr[0]).view(-1,1)
# x2=torch.linspace(-0.5,0.5,N_tr[1]).view(-1,1)
# x3=torch.linspace(0.5,6.28,N_tr[2]).view(-1,1)
# x_train = torch.cat( ( x1, x2, x3 ) ,0)
# y_train = torch.cat((torch.cos(x1).view(-1,1) + torch.randn_like(x1)*0.1 , torch.cos(x2).view(-1,1)+ torch.randn_like(x2)*3 , torch.cos(x3).view(-1,1)+ torch.randn_like(x3)*0.1),0)

# Test regression variance 1

N_tr = [50,50,50]
N_val = 400

#I define the test points and training points
x_val = torch.linspace(-10,10,N_val).view(-1,1)
#y_val = (x_val**2).view(-1,1)
y_val = torch.cos(x_val).view(-1,1)
x1=torch.linspace(-6.28,-2.10,N_tr[0]).view(-1,1)
x2=torch.linspace(-2.10,2.10,N_tr[1]).view(-1,1)
x3=torch.linspace(2.10,6.28,N_tr[2]).view(-1,1)
x_train = torch.cat( ( x1, x2, x3 ) ,0)
y_train = torch.cat((torch.cos(x1).view(-1,1) + torch.randn_like(x1) , torch.cos(x2).view(-1,1)+ torch.randn_like(x2) , torch.cos(x3).view(-1,1)+ torch.randn_like(x3)),0)


plt.figure(figsize=(10,5))
plt.plot(x_train.numpy(),y_train.numpy(),'.',markersize=30, label='x train')
plt.plot(x_val.numpy(),y_val.numpy(),'.',markersize=10, label='x test')

plt.legend(fontsize=20)
plt.show()

x_train = x_train.to(device)
y_train = y_train.to(device)

x_val = x_val.to(device)
y_val = y_val.to(device)

#Test function

# k = GPy.kern.RBF(input_dim = 1, variance = variance, lengthscale = lengthscale)
# C = k.K(x, x) + np.eye(no_points)*sig_noise**2
#
# y = np.random.multivariate_normal(np.zeros((no_points)), C)[:, None]
# y = (y - y.mean())
#
# # Training points
#
# #x_train = x[75:325]
# #y_train = y[75:325]
# x_train = x[range(1,400,10)]
# y_train = y[range(1,400,10)]

# Parameters of simulation

num_epochs, batch_size, nb_train = 2000, len(x_train), len(x_train)

# Definition of NN

net = MC_Dropout_Wrapper(input_dim = 1, output_dim=1, no_units=200, learn_rate=1e-2,
                         batch_size=batch_size, no_batches=1, init_log_noise=0, weight_decay=1e-2)

# Training (con fit)

for i in range(num_epochs):

    loss = net.fit(x_train, y_train)

    if i % 200 == 0:
        print('Epoch: %4d, Train loss = %7.3f, noise = %6.3f' % \
              (i, loss.cpu().data.numpy(), torch.exp(net.network.log_noise).cpu().data.numpy()))


samples = []
noises = []

# Testing (con forward)

for i in range(1000):
    preds = net.network.forward(x_val).cpu().data.numpy()
    samples.append(preds)

samples = np.array(samples)
means = (samples.mean(axis = 0)).reshape(-1)

# STD aleatoria
aleatoric = torch.exp(net.network.log_noise).cpu().data.numpy()
# STD empirica
epistemic = (samples.var(axis = 0)**0.5).reshape(-1)
# Distance between the 2 STDs
total_unc = (aleatoric**2 + epistemic**2)**0.5


c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# plt.figure(figsize = (6, 5))
# plt.style.use('default')
# plt.scatter(x_train, y_train, s = 10, marker = 'x', color = 'black', alpha = 0.5)
# plt.fill_between(np.linspace(-5, 5, 200), means + aleatoric, means + total_unc, color = c[0], alpha = 0.3, label = 'Epistemic + Aleatoric')
# plt.fill_between(np.linspace(-5, 5, 200), means - total_unc, means - aleatoric, color = c[0], alpha = 0.3)
# plt.fill_between(np.linspace(-5, 5, 200), means - aleatoric, means + aleatoric, color = c[1], alpha = 0.4, label = 'Aleatoric')
# plt.plot(np.linspace(-5, 5, 200), means, color = 'black', linewidth = 1)
# plt.xlim([-5, 5])
# plt.ylim([-5, 7])
# plt.xlabel('$x$', fontsize=30)
# plt.title('MC dropout', fontsize=40)
# plt.tick_params(labelsize=30)
# plt.xticks(np.arange(-4, 5, 2))
# plt.yticks(np.arange(-4, 7, 2))
# plt.gca().set_yticklabels([])
# plt.gca().yaxis.grid(alpha=0.3)
# plt.gca().xaxis.grid(alpha=0.3)
# plt.savefig('mc_dropout.pdf', bbox_inches = 'tight')
#
#
# plt.show()
#
#
plt.figure(figsize=(10,5))
plt.scatter(x_train, y_train, s = 10, marker = 'x', color = 'black', alpha = 0.5)
plt.plot(x_val.cpu().numpy(),samples[:].squeeze().T, 'C0',alpha=0.01)
# plt.plot(x_val, means, 'C1',alpha=0.9)
# plt.plot(x_val, means + aleatoric, 'C2',alpha=0.9)
# plt.plot(x_val, means + total_unc, 'C3',alpha=0.8,linewidth=3)
# plt.plot(x_val, means - aleatoric, 'C2',alpha=0.8,linewidth=3)
# plt.plot(x_val, means - total_unc, 'C3',alpha=0.8,linewidth=3)
# plt.savefig('mc_dropout_figo.pdf', bbox_inches = 'tight')

plt.plot(x_val.cpu().numpy(),samples.mean(0).squeeze().T, 'C1',alpha=0.9)
plt.plot(x_val.cpu().numpy(),samples.mean(0).squeeze().T +samples.std(0).squeeze().T, 'C1',alpha=0.8,linewidth=3)
plt.plot(x_val.cpu().numpy(),samples.mean(0).squeeze().T -samples.std(0).squeeze().T, 'C1',alpha=0.8,linewidth=3)

plt.plot(x_val.cpu().numpy(),y_val.cpu().numpy(),'r-',markersize=10 ,label='x test')




plt.ylim([-5,5])
plt.savefig('mc_dropout_no_out_var1.pdf', bbox_inches = 'tight')
plt.show()
