# -*- coding: utf-8 -*-
# Auto-generated from 'stats_sigdet_dPrime.ipynb' on 2025-08-08T15:22:56
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master statistics and machine learning: Intuition, Math, code
# ##### COURSE URL: udemy.com/course/statsml_x/?couponCode=202304 
# ## SECTION: Signal detection theory
# ### VIDEO: d-prime
# #### TEACHER: Mike X Cohen, sincxpress.com

# In [ ]
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# In [ ]
## example from the slides

# step 1
hitP = 22/30
faP  =  3/30

# step 2
hitZ = stats.norm.ppf(hitP)
faZ  = stats.norm.ppf(faP)

# step 3
dPrime = hitZ-faZ

print(dPrime)

# In [ ]
## failure scenarios and their resolutions

hitZ = stats.norm.ppf(0/30)
faZ  = stats.norm.ppf(22/30)

print(hitZ-faZ)

# In [ ]
## 2D d' space

# response probabilities
x  = np.arange(.01,1,.01)

# generate the space using tile expansion
dp = np.tile(stats.norm.ppf(x),(99,1)).T - np.tile(stats.norm.ppf(x),(99,1))


# show the 2D d' space
plt.imshow(dp,extent=[x[0],x[-1],x[0],x[-1]],origin='lower')
plt.xlabel('False alarm rate')
plt.ylabel('Hit rate')
plt.title("d'")
plt.colorbar()
plt.show()

