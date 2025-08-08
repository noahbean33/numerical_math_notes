# -*- coding: utf-8 -*-
# Auto-generated from 'stats_probtheory_cdfs.ipynb' on 2025-08-08T15:22:56
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master statistics and machine learning: Intuition, Math, code
# ##### COURSE URL: udemy.com/course/statsml_x/?couponCode=202304 
# ## SECTION: Probability theory
# ### VIDEO: cdf's and pdf's
# #### TEACHER: Mike X Cohen, sincxpress.com

# In [ ]
# import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# In [ ]
## example using log-normal distribution

# variable to evaluate the functions on
x = np.linspace(0,5,1001)

# note the function call pattern...
p1 = stats.lognorm.pdf(x,1)
c1 = stats.lognorm.cdf(x,1)

p2 = stats.lognorm.pdf(x,.1)
c2 = stats.lognorm.cdf(x,.1)

# In [ ]
# draw the pdfs
fig,ax = plt.subplots(2,1,figsize=(4,7))

ax[0].plot(x,p1/sum(p1)) # question: why divide by sum here?
ax[0].plot(x,p1/sum(p1), x,p2/sum(p2))
ax[0].set_ylabel('probability')
ax[0].set_title('pdf(x)')

# draw the cdfs
ax[1].plot(x,c1)
ax[1].plot(x,c1, x,c2)
ax[1].set_ylabel('probability')
ax[1].set_title('cdf(x)')
plt.show()

# In [ ]
## computing the cdf from the pdf

# compute the cdf
c1x = np.cumsum( p1*(x[1]-x[0]) )

plt.plot(x,c1)
plt.plot(x,c1x,'--')
plt.show()

