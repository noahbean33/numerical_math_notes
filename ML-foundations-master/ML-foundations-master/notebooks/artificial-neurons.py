# -*- coding: utf-8 -*-
# Auto-generated from 'artificial-neurons.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/artificial-neurons.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Artificial Neuron Layer

# %% [markdown]
# In this notebook, we use PyTorch tensors to create a layer of artificial neurons that could be used within a deep learning model architecture.

# In [ ]
import torch
import matplotlib.pyplot as plt

# In [ ]
_ = torch.manual_seed(42)

# %% [markdown]
# Set number of neurons: 

# In [ ]
n_input = 784 # Flattened 28x28-pixel image
n_dense = 128

# %% [markdown]
# Simulate an "input image" with a vector tensor `x`: 

# In [ ]
x = torch.rand(n_input) # Samples float values from [0,1) uniform distribution (interval doesn't include 1)

# In [ ]
x.shape

# In [ ]
x[0:6]

# In [ ]
_ = plt.hist(x)

# %% [markdown]
# Create tensors to store neuron parameters (i.e., weight matrix `W`, bias vector `b`) and initialize them with starting values: 

# In [ ]
b = torch.zeros(n_dense)

# In [ ]
b.shape

# In [ ]
b[0:6]

# In [ ]
W = torch.empty([n_input, n_dense])
W = torch.nn.init.xavier_normal_(W)

# In [ ]
W.shape

# In [ ]
W[0:4, 0:4]

# %% [markdown]
# Pass the "input image" `x` through a *dense* neuron layer with a *sigmoid activation function* to output the vector tensor `a`, which contains one element per neuron: 

# In [ ]
z = torch.add(torch.matmul(x, W), b)

# In [ ]
a = torch.sigmoid(z)

# In [ ]
a.shape

# In [ ]
_ = plt.hist(a)

# In [ ]

