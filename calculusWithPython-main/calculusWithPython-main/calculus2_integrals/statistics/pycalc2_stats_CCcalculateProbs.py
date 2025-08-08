# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_stats_CCcalculateProbs.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in statistics
# ### LECTURE: CodeChallenge: Calculating probabilities in python
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy import stats
import sympy.stats
import scipy.integrate as spi

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Exercise 1: Create a pdf and cdf

# In [ ]
z = np.linspace(-4,4,501)
dz = z[1]-z[0]

zval = 1

pdf = stats.norm.pdf(z) * dz
cdf = stats.norm.cdf(z)

_,axs = plt.subplots(2,1,figsize=(8,6))

axs[0].plot(z,pdf,'k')
axs[0].plot([zval,zval],[0,pdf[np.argmin(abs(z-zval))]],'k--',linewidth=1)
axs[0].fill_between(z[z>=zval],pdf[z>=zval],color='k',alpha=.2)
axs[0].set(xlim=z[[0,-1]],ylabel='Probability density',title='PDF')
axs[0].set_ylim(bottom=0)

axs[1].plot(z,cdf,'k')
axs[1].plot([zval,zval],[0,stats.norm.cdf(zval)],'k--',linewidth=1)
axs[1].plot([z[0],zval],np.full(2,stats.norm.cdf(zval)),'k--',linewidth=1)
axs[1].set(ylim=[-.02,1.02],xlim=z[[0,-1]],xlabel='z',ylabel='Cumulative probability',title='CDF')



plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Many ways to calculate the probability

# In [ ]
# Method 1: using numpy
p_fromNumpy = np.sum(pdf[np.argmin(abs(z-zval)):])
p_fromNumpy

# In [ ]
# Method 2: using spi
p_fromSpi = spi.simpson(pdf[np.argmin(abs(z-zval)):],dx=1)
p_fromSpi

# In [ ]
# Method 3: from cdf
p_fromCdf = 1 - cdf[np.argmin(abs(z-zval))]
p_fromCdf

# In [ ]
# Method 4: from stats.norm.cdf
p_fromStats = 1 - stats.norm.cdf(zval)
p_fromStats

# In [ ]
# Method 5: From sympy
x = sym.symbols('x')
norm = sym.stats.Normal('N',0,1)
symcdf = sym.stats.cdf(norm)(x)
p_fromSympy = symcdf.subs(x,sym.oo) - symcdf.subs(x,zval).evalf()
p_fromSympy

# In [ ]
# show all results
print(f'1) numpy: {p_fromNumpy:.8f}')
print(f'2) spi:   {p_fromSpi:.8f}')
print(f'3) cdf:   {p_fromCdf:.8f}')
print(f'4) stats: {p_fromStats:.8f}')
print(f'5) sympy: {p_fromSympy:.8f}')

# In [ ]

# %% [markdown]
# # Exercise 3: Probability of a finite interval

# In [ ]
zvalL = -2/3
zvalU = 1/3

_,axs = plt.subplots(2,1,figsize=(8,6))

axs[0].plot(z,pdf,'k')
axs[0].plot([zvalL,zvalL],[0,pdf[np.argmin(abs(z-zvalL))]],'k--',linewidth=1)
axs[0].plot([zvalU,zvalU],[0,pdf[np.argmin(abs(z-zvalU))]],'k--',linewidth=1)

z4fill = (z>=zvalL) & (z<=zvalU)
axs[0].fill_between(z[z4fill],pdf[z4fill],color='k',alpha=.2)
axs[0].set(xlim=z[[0,-1]],ylabel='Probability density',title='PDF')
axs[0].set_ylim(bottom=0)

axs[1].plot(z,cdf,'k')
axs[1].plot([zvalL,zvalL],[0,stats.norm.cdf(zvalL)],'k--',linewidth=1)
axs[1].plot([z[0],zvalL],np.full(2,stats.norm.cdf(zvalL)),'k--',linewidth=1)
axs[1].plot([zvalU,zvalU],[0,stats.norm.cdf(zvalU)],'k--',linewidth=1)
axs[1].plot([z[0],zvalU],np.full(2,stats.norm.cdf(zvalU)),'k--',linewidth=1)
axs[1].set(ylim=[-.02,1.02],xlim=z[[0,-1]],xlabel='z',ylabel='Cumulative probability',title='CDF')


plt.tight_layout()
plt.show()

# In [ ]
# Method 1: using numpy
p_fromNumpy = np.sum(pdf[z4fill])

# Method 2: using spi
p_fromSpi = spi.simpson(pdf[z4fill],dx=1)

# Method 3: from cdf
p_fromCdf = cdf[np.argmin(abs(z-zvalU))] - cdf[np.argmin(abs(z-zvalL))]

# Method 4: from stats.norm.cdf
p_fromStats = stats.norm.cdf(zvalU) - stats.norm.cdf(zvalL)

# Method 5: From sympy
p_fromSympy = symcdf.subs(x,zvalU) - symcdf.subs(x,zvalL).evalf()

# In [ ]
# show all results
print(f'1) numpy: {p_fromNumpy:.8f}')
print(f'2) spi:   {p_fromSpi:.8f}')
print(f'3) cdf:   {p_fromCdf:.8f}')
print(f'4) stats: {p_fromStats:.8f}')
print(f'5) sympy: {p_fromSympy:.8f}')

# In [ ]

# %% [markdown]
# # Exercise 4: With higher resolution

# In [ ]

