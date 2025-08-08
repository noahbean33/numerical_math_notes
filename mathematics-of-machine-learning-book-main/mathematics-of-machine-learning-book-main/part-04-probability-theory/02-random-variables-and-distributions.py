# -*- coding: utf-8 -*-
# Auto-generated from '02-random-variables-and-distributions.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # Chapter 19. Random variables and distributions

# %% [markdown]
# ## Discrete distributions

# %% [markdown]
# ### The Bernoulli distribution

# In [1]
from scipy.stats import bernoulli

# In [2]
[bernoulli.rvs(p=0.5) for _ in range(10)]    # ten Bernoulli(0.5)-distributed random numbers 

# In [3]
import matplotlib.pyplot as plt

params = [0.25, 0.5, 0.75]


with plt.style.context("seaborn-v0_8"):
    fig, axs = plt.subplots(1, len(params), figsize=(4*len(params), 4), sharey=True)
    fig.suptitle("The Bernoulli distribution")
    for ax, p in zip(axs, params):
        x = range(2)
        y = [bernoulli.pmf(k=k, p=p) for k in x]
        ax.bar(x, y)
        ax.set_title(f"p = {p}")
        ax.set_ylabel("P(X = k)")
        ax.set_xlabel("k")  
    plt.show()

# %% [markdown]
# ### The binomial distribution

# In [4]
from scipy.stats.distributions import binom


params = [(20, 0.25), (20, 0.5), (20, 0.75)]


with plt.style.context("seaborn-v0_8"):
    fig, axs = plt.subplots(1, len(params), figsize=(4*len(params), 4), sharey=True)
    fig.suptitle("The binomial distribution")
    for ax, (n, p) in zip(axs, params):
        x = range(n+1)
        y = [binom.pmf(n=n, p=p, k=k) for k in x]
        ax.bar(x, y)
        ax.set_title(f"n = {n}, p = {p}")
        ax.set_ylabel("P(X = k)")
        ax.set_xlabel("k")    

    plt.show()

# %% [markdown]
# ### The geometric distribution

# In [5]
from scipy.stats import geom


params = [0.2, 0.5, 0.8]


with plt.style.context("seaborn-v0_8"):
    fig, axs = plt.subplots(1, len(params), figsize=(5*len(params), 5), sharey=True)
    fig.suptitle("The geometric distribution")
    for ax, p in zip(axs, params):
        x = range(1, 20)
        y = [geom.pmf(p=p, k=k) for k in x]
        ax.bar(x, y)
        ax.set_title(f"p = {p}")
        ax.set_ylabel("P(X = k)")
        ax.set_xlabel("k")

    plt.show()

# %% [markdown]
# ### The uniform distribution

# In [6]
from scipy.stats import randint


with plt.style.context("seaborn-v0_8"):
    fig = plt.figure(figsize=(10, 5))
    plt.title("The uniform distribution")

    x = range(-1, 9)
    y = [randint.pmf(k=k, low=1, high=7) for k in x]
    plt.bar(x, y)
    plt.ylim(0, 1)
    plt.ylabel("P(X = k)")
    plt.xlabel("k")

    plt.show()

# %% [markdown]
# ### Sums of discrete random variables

# In [7]
import numpy as np


dist_1 = [0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
dist_2 = [0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
sum_dist = np.convolve(dist_1, dist_1)


with plt.style.context("seaborn-v0_8"):
    plt.figure(figsize=(10, 5))
    plt.bar(range(0, len(sum_dist)), sum_dist)
    plt.title("Distribution of X₁ + X₂")
    plt.ylabel("P(X + Y = k)")
    plt.xlabel("k")
    plt.show()

# %% [markdown]
# ## Real-valued distributions

# In [8]
from scipy.stats import uniform
X = np.linspace(-0.5, 1.5, 100)
y = uniform.cdf(X)


with plt.style.context('seaborn-v0_8'):
    plt.figure(figsize=(10, 5))
    plt.title("The uniform distribution")
    plt.plot(X, y)
    plt.show()

# %% [markdown]
# ### The exponential distribution

# In [9]
from scipy.stats import expon
X = np.linspace(-0.5, 10, 100)
params = [0.1, 1, 10]
ys = [expon.cdf(X, scale=1/l) for l in params]


with plt.style.context('seaborn-v0_8'):
    plt.figure(figsize=(10, 5))
    
    for l, y in zip(params, ys):
        plt.plot(X, y, label=f"λ = {l}")
    
    plt.title("The exponential distribution")
    plt.legend()
    plt.show()

# %% [markdown]
# ### The normal distribution

# In [10]
from scipy.stats import norm
X = np.linspace(-10, 10, 1000)
σs = [0.5, 1, 2, 3]
ys = [norm.pdf(X, scale=σ) for σ in σs]


with plt.style.context('seaborn-v0_8'):
    plt.figure(figsize=(10, 5))
    
    for σ, y in zip(σs, ys):
        plt.plot(X, y, label=f"σ = {σ}")
    
    plt.title("The bell curves")
    plt.legend()
    plt.show()

# In [11]
X = np.linspace(-10, 10, 1000)
σs = [0.5, 1, 2, 3]
ys = [norm.cdf(X, scale=σ) for σ in σs]


with plt.style.context('seaborn-v0_8'):
    plt.figure(figsize=(10, 5))
    
    for σ, y in zip(σs, ys):
        plt.plot(X, y, label=f"σ = {σ}")
    
    plt.title("The normal distribution")
    plt.legend()

    plt.show()

