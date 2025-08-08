# -*- coding: utf-8 -*-
# Auto-generated from 'learning-rate-scheduling.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/learning-rate-scheduling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Learning Rate Scheduling

# %% [markdown]
# This notebook improves upon the [*SGD from Scratch* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/SGD-from-scratch.ipynb) by: 
# 
# 1. Using efficient PyTorch `DataLoader()` iterable to batch data for SGD
# 2. Scheduling a variable learning rate

# In [1]
import torch
import matplotlib.pyplot as plt

# %% [markdown]
# ### Simulate data

# In [2]
_ = torch.manual_seed(42)

# In [3]
n = 8000000

# In [4]
x = torch.linspace(0., 8., n)

# %% [markdown]
# $$y = mx + b + \epsilon = -0.5x + 2 + \mathcal{N}(0, 1)$$

# In [5]
y = -0.5*x + 2 + torch.normal(mean=torch.zeros(n), std=1)

# %% [markdown]
# Randomly sample 2000 data points for model validation: 

# In [6]
from sklearn.model_selection import train_test_split

# In [7]
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.00025, random_state=42)

# In [8]
x_valid.shape

# In [9]
x_train.shape

# In [10]
fig, ax = plt.subplots()
plt.title("Clinical Trial")
plt.xlabel("Drug dosage (mL)")
plt.ylabel("Forgetfulness")
_ = ax.scatter(x_valid, y_valid, alpha=0.1)

# %% [markdown]
# ### Define model and "randomly" initialize model parameters

# In [11]
def regression(my_x, my_m, my_b):
    return my_m*my_x + my_b

# In [12]
m = torch.tensor([0.9]).requires_grad_()
b = torch.tensor([0.1]).requires_grad_()

# In [13]
fig, ax = plt.subplots()

ax.scatter(x_valid, y_valid, alpha=0.1)

x_min, x_max = ax.get_xlim()
y_min = regression(x_min, m, b).detach().item()
y_max = regression(x_max, m, b).detach().item()

plt.ylabel('b = {}'.format('%.3g' % b.item()))
plt.xlabel('m = {}'.format('%.3g' % m.item()))

ax.set_xlim([x_min, x_max])
_ = ax.plot([x_min, x_max], [y_min, y_max], c='C01')

# %% [markdown]
# ### Optimize parameters via SGD

# In [14]
from torch.utils.data import TensorDataset, DataLoader

# In [15]
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32) # set shuffle=True if multiple epochs

# In [16]
x_batch, y_batch = next(iter(train_loader))

# In [17]
x_batch

# In [18]
y_batch

# %% [markdown]
# **Step 1**: Forward pass

# In [19]
yhat = regression(x_batch, m, b)
yhat

# %% [markdown]
# **Step 2**: Compare $\hat{y}$ with true $y$ to calculate cost $C$

# In [20]
def mse(my_yhat, my_y): 
    sigma = torch.sum((my_yhat - my_y)**2)
    return sigma/len(my_y)

# In [21]
C = mse(yhat, y_batch)
C

# In [22]
def labeled_regression_plot(my_x, my_y, my_m, my_b, my_C, include_grad=True):
    
    title = 'Cost = {}'.format('%.3g' % my_C.item())
    if include_grad:
        xlabel = 'm = {}, m grad = {}'.format('%.3g' % my_m.item(), '%.3g' % my_m.grad.item())
        ylabel = 'b = {}, b grad = {}'.format('%.3g' % my_b.item(), '%.3g' % my_b.grad.item())
    else:
        xlabel = 'm = {}'.format('%.3g' % my_m.item())        
        ylabel = 'b = {}'.format('%.3g' % my_b.item())
    
    fig, ax = plt.subplots()
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    ax.scatter(my_x, my_y)
    
    x_min, x_max = ax.get_xlim()
    y_min = regression(x_min, my_m, my_b).detach().item()
    y_max = regression(x_max, my_m, my_b).detach().item()

    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max], c='C01')

# In [23]
labeled_regression_plot(x_batch, y_batch, m, b, C, include_grad=False)

# %% [markdown]
# **Step 3**: Use autodiff to calculate gradient of $C$ w.r.t. parameters

# In [24]
C.backward()

# In [25]
gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
gradient

# In [26]
theta = torch.tensor([[b, m]]).T 
theta

# %% [markdown]
# **Step 4**: Gradient descent $$ \boldsymbol{\theta}' = \boldsymbol{\theta} - \alpha \nabla C $$

# In [27]
lr = 0.02 # Doubled

# In [28]
new_theta = theta - lr*gradient
new_theta

# %% [markdown]
# Confirm $C$ is lower: 

# In [29]
b = new_theta[0]
m = new_theta[1]

# In [30]
C = mse(regression(x_batch, m, b), y_batch)
C

# %% [markdown]
# ### Rinse and Repeat

# In [31]
b.requires_grad_()
_ = m.requires_grad_() 

# %% [markdown]
# Set $\lambda$ hyperparameter for learning rate decay:

# In [32]
lambd = 0.995 # 'lambda' is a Python reserved term

# In [33]
lr = lambd * lr
lr

# In [34]
rounds = 1000 
for r in range(rounds): 
    
    x_batch, y_batch = next(iter(train_loader)) # Efficient batching
    
    yhat = regression(x_batch, m, b) # Step 1
    C = mse(yhat, y_batch) # Step 2
    
    C.backward() # Step 3
    
    if r % 50 == 0:
        print('Step {}, cost {}, m grad {}, b grad {}, lr {}'.format(r, '%.3g' % C.item(), '%.3g' % m.grad.item(), '%.3g' % b.grad.item(), '%.3g' % lr))
    
    gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
    theta = torch.tensor([[b, m]]).T 
    
    new_theta = theta - lr*gradient # Step 4
    
    b = new_theta[0].requires_grad_()
    m = new_theta[1].requires_grad_()
    
    lr = lambd * lr # Decay learning rate

# In [35]
labeled_regression_plot(x_batch, y_batch, m, b, C, include_grad=False)

# In [36]
validation_cost = mse(regression(x_valid, m, b), y_valid)

# In [37]
fig, ax = plt.subplots()

ax.scatter(x_valid, y_valid, alpha=0.1)

x_min, x_max = ax.get_xlim()
y_min = regression(x_min, m, b).detach().item()
y_max = regression(x_max, m, b).detach().item()

plt.title('Validation cost = {}'.format('%.3g' % validation_cost.item()))
plt.ylabel('b = {}'.format('%.3g' % b.item()))
plt.xlabel('m = {}'.format('%.3g' % m.item()))

ax.set_xlim([x_min, x_max])
_ = ax.plot([x_min, x_max], [y_min, y_max], c='C01')

# %% [markdown]
# Relative to the [*SGD from Scratch* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/SGD-from-scratch.ipynb): 
# 
# 1. Carry out 1k SGD rounds in fraction of time it took to do 100
# 2. Lower validation cost due to more SGD rounds and (independently) learning rate scheduling
# 
# (Note that because of $\epsilon$ in the data simulation step, reaching zero cost on training data would only be possible with extreme overfitting and this would not correspond to improved fit on the validation data.)

