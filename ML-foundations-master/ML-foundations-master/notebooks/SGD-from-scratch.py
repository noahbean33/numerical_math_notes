# -*- coding: utf-8 -*-
# Auto-generated from 'SGD-from-scratch.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/SGD-from-scratch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Stochastic Gradient Descent from Scratch

# %% [markdown]
# This notebook expands the upon the [*Gradient Descent from Scratch* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/gradient-descent-from-scratch.ipynb) to introduce *stochastic* gradient descent (SGD).

# In [ ]
import torch
import numpy as np
import matplotlib.pyplot as plt

# In [ ]
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ### Simulate data

# %% [markdown]
# Create a vector tensor `x` with eight million points spaced evenly from zero to eight:

# In [ ]
n = 8000000

# In [ ]
x = torch.linspace(0., 8., n) # using typical 'x' convention for vector instead of 'xs'

# %% [markdown]
# Use the same line equation as in the [*Regression in PyTorch* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/regression-in-pytorch.ipynb) to simulate eight million $y$ values for the vector `y`. That is, $m = -0.5$ and $b = 2$:
# $$y = mx + b + \epsilon = -0.5x + 2 + \mathcal{N}(0, 1)$$

# In [ ]
y = -0.5*x + 2 + torch.normal(mean=torch.zeros(n), std=1) # 'y' vector instead of 'ys'

# %% [markdown]
# Let's randomly sample a couple thousand points for the purpose of data visualization:

# In [6]
indices = np.random.choice(n, size=2000, replace=False)

# In [7]
fig, ax = plt.subplots()
plt.title("Clinical Trial")
plt.xlabel("Drug dosage (mL)")
plt.ylabel("Forgetfulness")
_ = ax.scatter(x[indices], y[indices], alpha=0.1)

# %% [markdown]
# ### Define model and "randomly" initialize model parameters

# In [8]
def regression(my_x, my_m, my_b):
    return my_m*my_x + my_b

# In [9]
m = torch.tensor([0.9]).requires_grad_()
b = torch.tensor([0.1]).requires_grad_()

# In [10]
fig, ax = plt.subplots()

ax.scatter(x[indices], y[indices], alpha=0.1)

x_min, x_max = ax.get_xlim()
y_min = regression(x_min, m, b).detach().numpy() # detach() stops requiring grad on Tensor (required before conversion to plottable Numpy array)
y_max = regression(x_max, m, b).detach().numpy()

plt.ylabel('b = {}'.format('%.3g' % b.item()))
plt.xlabel('m = {}'.format('%.3g' % m.item()))

ax.set_xlim([x_min, x_max])
_ = ax.plot([x_min, x_max], [y_min, y_max], c='C01')

# %% [markdown]
# ### Optimize parameters via SGD

# %% [markdown]
# Randomly (*stochastically*) sample a mini-batch of data for gradient descent:

# In [11]
batch_size = 32 # model hyperparameter

# %% [markdown]
# Sample without replacement. If you run out of data (which we won't in this example because we'll do 100 rounds of training so use 3200 data points, 0.04% of the 8m), commence another epoch with the full complement of data (we'll cover this in later NBs).

# In [12]
batch_indices = np.random.choice(n, size=batch_size, replace=False)

# In [13]
x[batch_indices]

# %% [markdown]
# Other than sampling a mini-batch, we optimize by following identical steps to those in the *Gradient Descent from Scratch* notebook:

# %% [markdown]
# **Step 1**: Forward pass

# In [14]
yhat = regression(x[batch_indices], m, b)
yhat

# %% [markdown]
# **Step 2**: Compare $\hat{y}$ with true $y$ to calculate cost $C$

# In [15]
def mse(my_yhat, my_y):
    sigma = torch.sum((my_yhat - my_y)**2)
    return sigma/len(my_y)

# In [16]
C = mse(yhat, y[batch_indices])
C

# In [17]
def labeled_regression_plot(my_x, my_y, my_m, my_b, my_C, include_grad=True):

    # Convert tensors to numpy arrays using detach():
    x_np = my_x.detach().numpy()
    y_np = my_y.detach().numpy()
    m_val = my_m.detach()
    b_val = my_b.detach()
    C_val = my_C.detach()

    title = 'Cost = {}'.format('%.3g' % C_val.item())
    if include_grad:
        xlabel = 'm = {}, m grad = {}'.format('%.3g' % m_val.item(), '%.3g' % my_m.grad.item())
        ylabel = 'b = {}, b grad = {}'.format('%.3g' % b_val.item(), '%.3g' % my_b.grad.item())
    else:
        xlabel = 'm = {}'.format('%.3g' % m_val.item())
        ylabel = 'b = {}'.format('%.3g' % b_val.item())

    fig, ax = plt.subplots()

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    ax.scatter(x_np, y_np)

    x_min, x_max = ax.get_xlim()
    y_min = regression(x_min, m_val, b_val)
    y_max = regression(x_max, m_val, b_val)

    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max], c='C01')

# In [18]
labeled_regression_plot(x[batch_indices], y[batch_indices], m, b, C, include_grad=False)

# %% [markdown]
# **Step 3**: Use autodiff to calculate gradient of $C$ w.r.t. parameters

# In [19]
C.backward()

# In [20]
m.grad

# In [21]
b.grad

# %% [markdown]
# The gradient of cost, $\nabla C$, is:

# In [22]
gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
gradient

# %% [markdown]
# The vector of parameters $\boldsymbol{\theta}$ is:

# In [23]
theta = torch.tensor([[b, m]]).T
theta

# %% [markdown]
# **Step 4**: Gradient descent $$ \boldsymbol{\theta}' = \boldsymbol{\theta} - \alpha \nabla C $$

# In [24]
lr = 0.01

# In [25]
new_theta = theta - lr*gradient
new_theta

# %% [markdown]
# That's it! Let's update the `m` and `b` variables and confirm the correspond to a lower cost $C$:

# In [26]
b = new_theta[0]
m = new_theta[1]

# In [27]
C = mse(regression(x[batch_indices], m, b), y[batch_indices])
C

# %% [markdown]
# ### Rinse and Repeat

# In [28]
b.requires_grad_()
_ = m.requires_grad_()

# %% [markdown]
# Instead of looping through epochs, we'll loop through rounds of SGD:

# In [29]
rounds = 100 # Use additional rounds (e.g., 1000) for better fit (or use a "fancy" optimizer)
for r in range(rounds):

    # This sampling step is slow; we'll cover much quicker batch sampling later:
    batch_indices = np.random.choice(n, size=batch_size, replace=False)

    yhat = regression(x[batch_indices], m, b) # Step 1
    C = mse(yhat, y[batch_indices]) # Step 2

    C.backward() # Step 3

    if r % 10 == 0:
        print('Step {}, cost {}, m grad {}, b grad {}'.format(r, '%.3g' % C.item(), '%.3g' % m.grad.item(), '%.3g' % b.grad.item()))

    gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
    theta = torch.tensor([[b, m]]).T

    new_theta = theta - lr*gradient # Step 4

    b = new_theta[0].requires_grad_()
    m = new_theta[1].requires_grad_()

# In [30]
labeled_regression_plot(x[batch_indices], y[batch_indices], m, b, C, include_grad=False)

# %% [markdown]
# Since we have so many data points and we were sampling without replacement throughout this notebook, we can use our initial `indices` sample as model *validation data*:

# In [31]
validation_cost = mse(regression(x[indices], m, b), y[indices])

# In [32]
# Detach and convert tensors to numpy arrays for plotting:
x_np = x[indices].detach().numpy()
y_np = y[indices].detach().numpy()
m_val = m.detach()
b_val = b.detach()
validation_cost_val = validation_cost.detach()

fig, ax = plt.subplots()

ax.scatter(x_np, y_np, alpha=0.1)

x_min, x_max = ax.get_xlim()
y_min = regression(x_min, m_val, b_val)
y_max = regression(x_max, m_val, b_val)

plt.title('Validation cost = {}'.format('%.3g' % validation_cost_val.item()))
plt.ylabel('b = {}'.format('%.3g' % b_val.item()))
plt.xlabel('m = {}'.format('%.3g' % m_val.item()))

ax.set_xlim([x_min, x_max])
_ = ax.plot([x_min, x_max], [y_min, y_max], c='C01')

# %% [markdown]
# The model could fit the validation data better by being run for a thousand rounds instead of a hundred. However, using the batch-sampling method in this notebook is painfully slow. See the [*Learning Rate Scheduling* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/learning-rate-scheduling.ipynb), which builds on what we've done here to both more efficiently sample batches and refine the learning rate, thereby resulting in a much better-fitting model in a small fraction of the time.

