# Bayesian-Deep-Learning-Clarotto-Franchini-Lamperti
In this project we aim at comparing a Bayesian Neural Network trained with Hamiltonian Monte Carlo and a MC Dropout Neural Network focusing on regression and reinforcement learning problem. We used the library Pytorch.


## Table of contents
* [Installation guide](#installation-guide)
* [The architecture of the network](#The-architecture-of-the-network)
* [Comparison of results](#Comparison-of-results)
* [Reinforcement Learning](#Reinforcement-Learning)
* [Inspiration](#inspiration)
* [Contact](#contact)

## Installation guide

In the _bayes_reg.py_ implementation we used an external package called  *hamiltorch* which we used to train the NN with the Hamiltonian MC method. To install it you have to execute the following commands:

git clone https://github.com/AdamCobb/hamiltorch.git

```python

import foobar

cd hamiltorch

pip install .
```



## The architecture of the network

The process of coding and analysing the Bayesian Neural Network and the Neural Network with Dropout is composed of different steps:

1. Train theBayesianNN with Hamiltonian MonteCarlo with the library Hamiltorch;
2. Train the NN with dropout with homoscedastic and heteroscedastic dropout (p = 0.7, i.e 30% of missing nodes); 
3. Find the architecture of the Neural Network in which each method works best;
4. Test the two methods on the same function 
 
                                     f (x) = sin(3(x + 2) − (x + 2)^2) + exp(−0.2x)


We choose to maintain the same number of layers (2 hidden layers) and the same activation functions (_sin_ for the first layer, _ReLU_ for the second layer).

The only variable that changes between the two nets is the number of nodes in each layer. We aim at using two nets which give the results in almost the same time. The architecture that makes the duration of the training similar is a Bayesian NN with 5 nodes in each layer and an NN with Dropout with a much larger number of nodes in each layer (300 nodes). We must consider that a part of those are switched off during each forward pass (only 70% of them contribute to the network).

## Comparison of results

**Homoscedastic and heteroscedastic residuals**

In the Neural Network with dropout, it’s necessary to choose which type of residuals must be used. We test the network with both homoscedastic residuals and heteroscedastic residuals.

If we use homoscedastic residuals, we fix the variance for all data to  σ= 0.3, so that the training data have the exatly same design as in the bayesian NN. Instead, if we choose to have heteroscedastic residuals, the variance depends on the variability of data. The residuals are learnt during the training process.


If we look at the results of the neural network, we see that the uncertainty varies a lot in the two cases. When the dropout is applied with homoscedastic residuals, the uncertainty is underestimated, especially when the function becomes periodic. When the residuals are heteroscedastic and are learnt during the training phase, we remark a higher uncertainty where the function is periodic.


<p float="left">
  <img src="images/mc_homoscedastic.png" width="400" />
  <img src="images/mc_heteroscedastic.png" width="400" />
</p>


**Fitness and uncertainty - Bayesian NN vs. Het- eroscedastic dropout**

We look at the goodness of the fit and at the uncertainty given by the credibility intervals computed on the samples of the neural network.
The Bayesian NN gives good results. The function is well learnt and the uncer- tainty is pretty high, especially outside the training area. The MC dropout gives almost perfect predictions and null uncertainty in U(x = 0), but underestimates the uncertainty out of the central more linear part.


<p float="left">
  <img src="images/hamiltonian_bayes.png" width="400" />
  <img src="images/mc_dropout_heteroscedastic.png" width="400" />
</p>

**Fitness and uncertainty - Bayesian NN vs. Het- eroscedastic dropout**

The last comparison we do is about the resilience of the networks when some outliers are added. We aim at seeing what is the method that better works when some wrong data form the training dataset. We test the two networks with 8 outliers inserted be- tween 0.5 and 2, all randomly selected around y = 2 with σ = 0.3.


<p float="left">
  <img src="images/bayesian_outilers.png" width="400" />
  <img src="images/mc_dropout_outliers.png" width="400" />
</p>

From the simulations we remark that the Bayesian Neural Network tends to follow the outliers and the entire curve be higher in the area where they have been added.

## Reinforcement Learning

We try to train and test the cartpole problem, through the two different methods: ε- greedy algorithm and implicit Thompson sampling (via dropout). We discuss only the dropout approach, since it’s the one which deals with a Bayesian approach.

Thanks to the introduction of this sampling procedure, we can analyze the variance of choosing to go right or left at each passage from one state to the next one. In this way we may detect the states where the cartpole is more sure about its following decision and the ones where the uncertainty is higher.

## Inspiration
Adam Cobb https://adamcobb.github.io 

Javer Antoran https://github.com/JavierAntoran/Bayesian-Neural-Networks

Documentation on Q-learning  https://github.com/hzxsnczpku/banrinochoujou/blob/master/doc/hw.pdf

## Contact


