# -*- coding: utf-8 -*-
# Auto-generated from '01-what-is-probability.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # Chapter 18. What is probability?

# %% [markdown]
# ## The axioms of probability

# %% [markdown]
# ### How to interpret probability?

# In [1]
import numpy as np
from scipy.stats import randint


n_tosses = 1000
# coin tosses: 0 for tails and 1 for heads
coin_tosses = [randint.rvs(low=0, high=2) for _ in range(n_tosses)]
averages = [np.mean(coin_tosses[:k+1]) for k in range(n_tosses)]

# In [2]
import matplotlib.pyplot as plt


with plt.style.context("seaborn-v0_8"):
    plt.figure(figsize=(16, 8))
    plt.title("Relative frequency of the coin tosses")
    plt.xlabel("Number of tosses")
    plt.ylabel("Relative frequency")
    
    # plotting the averages
    plt.plot(range(n_tosses), averages, linewidth=3) # the averages
    
    # plotting the true expected value
    plt.plot([-100, n_tosses+100], [0.5, 0.5], c="k")
    plt.xlim(-10, n_tosses+10)  
    plt.ylim(0, 1)
    plt.show()

