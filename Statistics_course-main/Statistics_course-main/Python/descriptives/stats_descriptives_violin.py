# -*- coding: utf-8 -*-
# Auto-generated from 'stats_descriptives_violin.ipynb' on 2025-08-08T15:22:56
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master statistics and machine learning: Intuition, Math, code
# ##### COURSE URL: udemy.com/course/statsml_x/?couponCode=202304 
# ## SECTION: Descriptive statistics
# ### VIDEO: Violin plots
# #### TEACHER: Mike X Cohen, sincxpress.com

# In [ ]
# import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# In [ ]
## create the data

n = 1000
thresh = 5 # threshold for cropping data

data = np.exp( np.random.randn(n) )
data[data>thresh] = thresh + np.random.randn(sum(data>thresh))*.1

# show histogram
plt.hist(data,30)
plt.title('Histogram')
plt.show()

# show violin plot
plt.violinplot(data)
plt.title('Violin')
plt.show()

# In [ ]
# another option: swarm plot

import seaborn as sns
sns.swarmplot(data,orient='v')

