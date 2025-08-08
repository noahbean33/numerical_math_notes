# -*- coding: utf-8 -*-
# Auto-generated from 'single-point-regression-gradient.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/single-point-regression-gradient.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Gradient of a Single-Point Regression

# %% [markdown]
# In this notebook, we calculate the gradient of quadratic cost with respect to a straight-line regression model's parameters. We keep the partial derivatives as simple as possible by limiting the model to handling a single data point. 

# In [1]
import torch

# %% [markdown]
# Let's use the same data as we did in the [*Regression in PyTorch* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/regression-in-pytorch.ipynb) as well as for demonstrating the Moore-Penrose Pseudoinverse in the [*Linear Algebra II* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/2-linear-algebra-ii.ipynb):

# In [2]
xs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])

# In [3]
ys = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])

# %% [markdown]
# The slope of a line is given by $y = mx + b$:

# In [4]
def regression(my_x, my_m, my_b):
    return my_m*my_x + my_b

# %% [markdown]
# Let's initialize $m$ and $b$ with the same "random" near-zero values as we did in the *Regression in PyTorch* notebook: 

# In [5]
m = torch.tensor([0.9]).requires_grad_()

# In [6]
b = torch.tensor([0.1]).requires_grad_()

# %% [markdown]
# To keep the partial derivatives as simple as possible, let's move forward with a single instance $i$ from the eight possible data points: 

# In [7]
i = 7
x = xs[i]
y = ys[i]

# In [8]
x

# In [9]
y

# %% [markdown]
# **Step 1**: Forward pass

# %% [markdown]
# We can flow the scalar tensor $x$ through our regression model to produce $\hat{y}$, an estimate of $y$. Prior to any model training, this is an arbitrary estimate:

# In [10]
yhat = regression(x, m, b)
yhat

# %% [markdown]
# **Step 2**: Compare $\hat{y}$ with true $y$ to calculate cost $C$

# %% [markdown]
# In the *Regression in PyTorch* notebook, we used mean-squared error, which averages quadratic cost over multiple data points. With a single data point, here we can use quadratic cost alone. It is defined by: $$ C = (\hat{y} - y)^2 $$

# In [11]
def squared_error(my_yhat, my_y):
    return (my_yhat - my_y)**2

# In [12]
C = squared_error(yhat, y)
C

# %% [markdown]
# **Step 3**: Use autodiff to calculate gradient of $C$ w.r.t. parameters

# In [13]
C.backward()

# %% [markdown]
# The partial derivative of $C$ with respect to $m$ ($\frac{\partial C}{\partial m}$) is: 

# In [14]
m.grad

# %% [markdown]
# And the partial derivative of $C$ with respect to $b$ ($\frac{\partial C}{\partial b}$) is: 

# In [15]
b.grad

# %% [markdown]
# **Return to *Calculus II* slides here to derive $\frac{\partial C}{\partial m}$ and $\frac{\partial C}{\partial b}$.**

# %% [markdown]
# $$ \frac{\partial C}{\partial m} = 2x(\hat{y} - y) $$

# In [16]
2*x*(yhat.item()-y)

# %% [markdown]
# $$ \frac{\partial C}{\partial b} = 2(\hat{y}-y) $$

# In [17]
2*(yhat.item()-y)

# %% [markdown]
# ### The Gradient of Cost, $\nabla C$

# %% [markdown]
# The gradient of cost, which is symbolized $\nabla C$ (pronounced "nabla C"), is a vector of all the partial derivatives of $C$ with respect to each of the individual model parameters: 

# %% [markdown]
# $\nabla C = \nabla_p C = \left[ \frac{\partial{C}}{\partial{p_1}}, \frac{\partial{C}}{\partial{p_2}}, \cdots, \frac{\partial{C}}{\partial{p_n}} \right]^T $

# %% [markdown]
# In this case, there are only two parameters, $b$ and $m$: 

# %% [markdown]
# $\nabla C = \left[ \frac{\partial{C}}{\partial{b}}, \frac{\partial{C}}{\partial{m}} \right]^T $

# In [18]
gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
gradient

