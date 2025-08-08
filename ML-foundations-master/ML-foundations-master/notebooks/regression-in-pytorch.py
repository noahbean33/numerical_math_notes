# -*- coding: utf-8 -*-
# Auto-generated from 'regression-in-pytorch.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/regression-in-pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Regression in PyTorch

# %% [markdown]
# In this notebook, we use the PyTorch **automatic differentiation** library to fit a straight line to data points. Thus, here we use calculus to solve the same regression problem that we used the Moore-Penrose Pseudoinverse to solve in the [*Linear Algebra II* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/2-linear-algebra-ii.ipynb).

# In [1]
import torch
import matplotlib.pyplot as plt

# In [2]
x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.]) # E.g.: Dosage of drug for treating Alzheimer's disease
x

# %% [markdown]
# The $y$ values were created using the equation of a line $y = mx + b$. This way, we know what the model parameters to be learned are, say, $m = -0.5$ and $b = 2$. Random, normally-distributed noise has been added to simulate sampling error: 

# In [3]
# y = -0.5*x + 2 + torch.normal(mean=torch.zeros(8), std=0.2)

# %% [markdown]
# For reproducibility of this demo, here's a fixed example of $y$ values obtained by running the commented-out line above: 

# In [4]
y = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37]) # E.g.: Patient's "forgetfulness score"
y

# In [5]
fig, ax = plt.subplots()
plt.title("Clinical Trial")
plt.xlabel("Drug dosage (mL)")
plt.ylabel("Forgetfulness")
_ = ax.scatter(x, y)

# %% [markdown]
# Initialize the slope parameter $m$ with a "random" value of 0.9...

# %% [markdown]
# (**N.B.**: In this simple demo, we could guess approximately-correct parameter values to start with. Or, we could use an algebraic (e.g., Moore-Penrose pseudoinverse) or statistical (e.g., ordinary-least-squares regression) to solve for the parameters quickly. This tiny machine learning demo with two parameters and eight data points scales, however, to millions of parameters and millions of data points. The other approaches -- guessing, algebra, statistics -- do not come close to scaling in this way.)

# In [6]
m = torch.tensor([0.9]).requires_grad_()
m

# %% [markdown]
# ...and do the same for the $y$-intercept parameter $b$: 

# In [7]
b = torch.tensor([0.1]).requires_grad_()
b

# In [8]
def regression(my_x, my_m, my_b):
    return my_m*my_x + my_b

# In [9]
def regression_plot(my_x, my_y, my_m, my_b):
    
    fig, ax = plt.subplots()

    ax.scatter(my_x, my_y)
    
    x_min, x_max = ax.get_xlim()
    y_min = regression(x_min, my_m, my_b).detach().item()
    y_max = regression(x_max, my_m, my_b).detach().item()
    
    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max])

# In [10]
regression_plot(x, y, m, b)

# %% [markdown]
# **Return to slides here if following *Calculus I* class.**

# %% [markdown]
# ### Machine Learning
# In four easy steps :)

# %% [markdown]
# **Step 1**: Forward pass

# In [11]
yhat = regression(x, m, b)
yhat

# %% [markdown]
# **Step 2**: Compare $\hat{y}$ with true $y$ to calculate cost $C$

# %% [markdown]
# There is a PyTorch `MSELoss` method, but let's define it outselves to see how it works. MSE cost is defined by: $$C = \frac{1}{n} \sum_{i=1}^n (\hat{y_i}-y_i)^2 $$

# In [12]
def mse(my_yhat, my_y): 
    sigma = torch.sum((my_yhat - my_y)**2)
    return sigma/len(my_y)

# In [13]
C = mse(yhat, y)
C

# %% [markdown]
# **Step 3**: Use autodiff to calculate gradient of $C$ w.r.t. parameters

# In [14]
C.backward()

# In [15]
m.grad

# In [16]
b.grad

# %% [markdown]
# **Step 4**: Gradient descent

# In [17]
optimizer = torch.optim.SGD([m, b], lr=0.01)

# In [18]
optimizer.step()

# %% [markdown]
# Confirm parameters have been adjusted sensibly: 

# In [19]
m

# In [20]
b

# In [21]
regression_plot(x, y, m, b)

# %% [markdown]
# We can repeat steps 1 and 2 to confirm cost has decreased: 

# In [22]
C = mse(regression(x, m, b), y)
C

# %% [markdown]
# Put the 4 steps in a loop to iteratively minimize cost toward zero: 

# In [23]
epochs = 1000
for epoch in range(epochs):
    
    optimizer.zero_grad() # Reset gradients to zero; else they accumulate
    
    yhat = regression(x, m, b) # Step 1
    C = mse(yhat, y) # Step 2
    
    C.backward() # Step 3
    optimizer.step() # Step 4
    
    print('Epoch {}, cost {}, m grad {}, b grad {}'.format(epoch, '%.3g' % C.item(), '%.3g' % m.grad.item(), '%.3g' % b.grad.item()))

# In [24]
regression_plot(x, y, m, b)

# In [25]
m.item()

# In [26]
b.item()

# %% [markdown]
# **N.B.**: The model doesn't perfectly approximate the slope (-0.5) and $y$-intercept (2.0) used to simulate the outcomes $y$ at the top of this notebook. This reflects the imperfectness of the sample of eight data points due to adding random noise during the simulation step. In the real world, the best solution would be to sample additional data points: The more data we sample, the more accurate our estimates of the true underlying parameters will be. 

