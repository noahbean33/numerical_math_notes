# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_integrate_calculartSineLines.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integrating functions
# ### LECTURE: Calculart! Whispy sine lines
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import Math
import scipy.integrate as spi


# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# In [ ]
# x-axis grid
xx = np.linspace(-2*np.pi,2*np.pi,345)
dx = xx[1]-xx[0]

# define alpha values
alphas = np.linspace(1,2,20)

# initialize a figure
fig,axs = plt.subplots(1,3,figsize=(15,4),facecolor='#212121')

# loop over alpha values
for a in alphas:

  # create the function (uncomment one of these)
  f = (1-a)+np.cos(a*xx)
  # f = ( (1-a)+np.cos(a*xx) ) * np.exp(-xx**2/(np.pi**2))
  # f = (1-a)+np.cos(a*xx) + np.log(np.abs(xx)+.001)

  # numpy approximation of the integral
  fi = np.cumsum(f)*dx

  # scipy's version (can be more accurate for tricky integrals)
  fi2 = spi.cumulative_trapezoid(f,dx=dx,initial=0)

  # line color
  c = .1 + (a-1)*.8

  # plot the lines!
  axs[0].plot(xx,f,color=[c,c,c])
  axs[1].plot(xx,a+fi,color=[c,c,c])
  axs[2].plot(f,fi,color=[c,c,c])


# final adjustments
for a in axs:
  a.axis('off')
plt.tight_layout()
plt.show()

# In [ ]

