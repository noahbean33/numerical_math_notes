# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_intuition_CCgeomApprox.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration and applications
# ## SECTION: Intuition for integration
# ### LECTURE: CodeChallenge: Geometric approximations
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202506

# In [ ]
import numpy as np
import matplotlib.pyplot as plt

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
#  # Exercise 1: Try two different dx's

# In [ ]
# function for the function
def fx(u):
  return u**2 - .5

# In [ ]
# define two dx's
dxs = [ .2, .05 ]

# plot the function
_,axs = plt.subplots(1,2,figsize=(12,5))

for idx,dx in enumerate(dxs):

  # define the resolution
  xx = np.arange(-.5,1+dx/2,dx)

  # define the function
  y = fx(xx)

  # plot the function
  axs[idx].plot(xx,y,'ks-',linewidth=2,markersize=10,markerfacecolor=[.7,.3,.9])



  # initialize area
  area = 0

  # plot rectangles
  for xi in xx:

    # draw the rectangle
    axs[idx].fill_between([xi-dx/2,xi+dx/2],[fx(xi),fx(xi)],edgecolor='k',facecolor=[.9,.8,.9])

    # sum the area
    area += fx(xi)*dx

  # set the labels (*after* the for-loop)
  axs[idx].set(xlabel='x',ylabel=r'$y = x^2-.5$',title=r'Area = %.3f when $\Delta$x=%g' %(area,dx))


# finalize plot
plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: A range of dx's

# In [ ]
# resolutions
dxs = np.logspace(np.log10(.5),np.log10(.001),20)
areas = np.zeros(len(dxs))

# function bounds
bounds = [-1,1]

# loop over resolutions
for i,dx in enumerate(dxs):

  # x-axis grid
  xx = np.arange(bounds[0],bounds[1]+dx,dx)

  # compute area using rectangle area formula
  area = 0
  for xi in xx: area += fx(xi)*dx

  # store final result
  areas[i] = area

# In [ ]
_,ax = plt.subplots(1,figsize=(10,5))

# plot the results
ax.plot(dxs,areas,'ks-',linewidth=2,markerfacecolor='w',markersize=10)
ax.axvline(.2, linestyle='--',color='gray',zorder=-1)
ax.axvline(.05,linestyle='--',color='gray',zorder=-1)

ax.invert_xaxis()
ax.set_xscale('log')
ax.set(xlabel=r'$\Delta x$',ylabel='Area (estimate of definite integral)')
plt.show()

# In [ ]

