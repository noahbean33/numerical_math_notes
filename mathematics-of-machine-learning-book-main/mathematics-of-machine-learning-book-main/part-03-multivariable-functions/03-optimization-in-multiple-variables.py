# -*- coding: utf-8 -*-
# Auto-generated from '03-optimization-in-multiple-variables.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # Chapter 17. Optimization in multiple variables

# %% [markdown]
# ## Multivariable functions in code

# In [1]
class MultivariableFunction:
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        pass
    
    def grad(self):
        pass
    
    def parameters(self):
        return dict()

# In [2]
import numpy as np


class SquaredNorm(MultivariableFunction):
    def __call__(self, x: np.array):
        return np.sum(x**2)
    
    def grad(self, x: np.array):
        return 2*x

# In [3]
class Linear(MultivariableFunction):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
    
    def __call__(self, x: np.array):
        """
        x: np.array of shape (2, )
        """
        x = x.reshape(2)
        return self.a*x[0] + self.b*x[1]
    
    def grad(self, x: np.array):
        return np.array([self.a, self.b]).reshape(2, 1)
    
    def parameters(self):
        return {"a": self.a, "b": self.b}

# In [4]
g = Linear(a=1, b=-1)

g(np.array([1, 0]))

# %% [markdown]
# ## Gradient descent in its full form

# In [5]
def gradient_descent(
    f: MultivariableFunction,  
    x_init: np.array,               # the initial guess
    learning_rate: float = 0.1,     # the learning rate
    n_iter: int = 1000,             # number of steps
):   
    x = x_init
    
    for n in range(n_iter):
        grad = f.grad(x)
        x = x - learning_rate*grad
    
    return x

# In [6]
squared_norm = SquaredNorm()
local_minimum = gradient_descent(
    f=squared_norm, 
    x_init=np.array([10.0, -15.0])
)

# In [7]
local_minimum

