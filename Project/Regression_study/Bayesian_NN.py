##############################  BAYESIAN DEEP LEARNING  ############################## 
############### Clarotto Lucia, Franchini Alessandro, Lamperti Letizia ############### 

###############               BAYESIAN NEURAL NETWORK                  ###############
###############              Regression Neural Network                 ###############
###############    Plots of samples, mean and credibility intervals    ###############



######### LIBRARIES #########

import os
import numpy as np
import pymc3
import hamiltorch
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
hamiltorch.set_random_seed(123)

# Working only on cpu (NO CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path for plots
my_path = os.path.dirname(__file__) + '/img/'



######### CLASSES #########

# Neural network
class Net(nn.Module):

    def __init__(self, layer_sizes, loss = 'multi_class', bias=True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.loss = loss
        self.bias = bias
        
        # 3 hidden layers
        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1],bias=True)
        self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2],bias = self.bias)
        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3],bias = self.bias)


    def forward(self, x):   
        
        ## LAYER 1 (from input)
        x = self.l1(x)
        
        # activation function (sin)
        x = torch.sin(x)
        
        ## LAYER 2 (from 1 to 3)
        x = self.l2(x)
        
        # activation function (relu)
        x = torch.relu(x)
        
        ## LAYER 3 (to output)
        x = self.l3(x)

        return x



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
y_train = torch.cat((my_function(x1).view(-1,1) + torch.randn_like(x1)*0.3, my_function(x2).view(-1,1)+ torch.randn_like(x2)*0.3, my_function(x3).view(-1,1)+ torch.randn_like(x3)*0.3, my_function(x4).view(-1,1) + 3 + torch.randn_like(x4)+0.3),0)

plt.figure(figsize=(10,5))
plt.plot(x_train.numpy(),y_train.numpy(),'.',markersize=10, label='x train', alpha=0.6)
plt.plot(x_val.numpy(),y_val.numpy(),'-',linewidth=7, label='x test')
plt.legend(fontsize=20)
plt.title('Regression function')
plt.show()


x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)



# Hamiltonian parameters
step_size = 0.012
num_samples = 200
L = 10
tau_out = 100. #10.

# Number of neurons for every layer
layer_sizes = [1,5,5,1]


hamiltorch.set_random_seed(123)

# Definition of NN
net = Net(layer_sizes, loss='regression')

# Parameters of simulation
params_init = hamiltorch.util.flatten(net).to(device).clone()
print('Parameter size: ',params_init.shape[0])

tau_list = []
tau = .1#.1#.0001#/1000#/10#/10. # 1/10
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)


# Training of the model
params_hmc = hamiltorch.sample_model(net, x_train, y_train, model_loss='regression',params_init=params_init, num_samples=num_samples, step_size=step_size, num_steps_per_sample=L,tau_out=tau_out,normalizing_const=N_tr, tau_list=tau_list)

# Prediction
pred_list, log_prob_list = hamiltorch.predict_model(net, x_val, y_val, model_loss='regression', samples=params_hmc[:], tau_out=tau_out, tau_list=tau_list)

print(tau_list[0])
print(tau_out)
print('\nExpected validation log probability: {:.2f}'.format(torch.stack(log_prob_list).mean()))
print('\nExpected MSE: {:.2f}'.format(((pred_list.mean(0) - y_val)**2).mean()))

# Credibility intervals
CI=np.zeros((len(x_val),2))
for i in range(len(x_val)):
    CI[i,:]=pymc3.stats.hpd(pred_list[:,i].numpy())
# credible_interval=0.94, circular=False



######### PLOTS #########
    
plt.figure(figsize=(10,5))
plt.plot(x_val.cpu().numpy(),pred_list[:].cpu().numpy().squeeze().T, 'steelblue',alpha=0.01)
plt.plot(x_train[:(len(x_train)-n_outliers)],y_train[:(len(x_train)-n_outliers)],'r.',markersize=4, label='Train data',alpha=0.8)

### If there's outliers uncomment the following line
#plt.plot(x_train[(len(x_train)-n_outliers):],y_train[(len(x_train)-n_outliers):],'rx',markersize=6, label='Extra perturbated train data',alpha=0.8)

plt.plot(x_val.cpu().numpy(),pred_list.mean(0).cpu().numpy().squeeze().T, 'orange',alpha=1,linewidth=3,label='Posterior mean')
plt.plot(x_val.cpu().numpy(),CI[:,0].squeeze().T, 'darkorange', alpha=1, label='Credibility Interval')
plt.plot(x_val.cpu().numpy(),CI[:,1].squeeze().T, 'darkorange', alpha=1)
plt.plot(x_val.cpu().numpy(),y_val.cpu().numpy(),'lime',linewidth=2,alpha=0.6,label='Real function')
plt.ylim([-2,5])
plt.xlim([-6.5,6.5])
plt.legend(loc='upper right')

plt.title('Hamiltonian Bayesian Neural Network')
### If there's outliers uncomment the following line and comment the previous
#plt.title('Hamiltonian Bayesian Neural Network with Outliers')

### If you want to save the plots in the current path 
#plt.savefig(my_path+'Hamiltonian.pdf', bbox_inches = 'tight')
#plt.savefig(my_path+'Hamiltonian_outliers.pdf', bbox_inches = 'tight')
plt.show()