# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_intuition_CCdiscreteInts.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration and applications
# ## SECTION: Intuition for integration
# ### LECTURE: CodeChallenge: Drawing discrete integrals
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
# # Exercise 1: Functions to compute and plot the integral

# In [ ]
# create a function that computes and outputs the derivative and integral
def derivAndIntegral(x,fx):

  # difference (discrete derivative)
  dx = x[1] - x[0]
  df = np.diff(fx) / dx

  # cumulative sum (discrete integral)
  idf = np.cumsum(df) * dx

  # normalize the integral
  zeroIdx = np.argmin(abs(x-0)) # x-axis coordinate of x=0
  idf -= idf[zeroIdx] # set idf(0)=0
  idf += fx[zeroIdx]  # then add constant from original function

  # return the calculations
  return df,idf


# and a function that does the plotting
def plotTheFunctions():
  _,axs = plt.subplots(1,3,figsize=(12,4))

  # visualize the function
  axs[0].plot(x,fx,'ks',markerfacecolor='w',markersize=10,linewidth=2,alpha=.5)
  axs[0].set(xlabel='x',ylabel='y = f(x)',title='Original function')

  # visualize the derivative
  axs[1].plot(x[1:],df,'ks',markerfacecolor='w',markersize=10,linewidth=2,alpha=.5)
  axs[1].set(xlabel='x',ylabel='dy/dx',title='Discrete derivative')

  # visualize the integral
  axs[2].plot(x[1:],idf,'ks',markerfacecolor='w',markersize=10,linewidth=2,label='Integral approx.',alpha=.5)

  # and plot the original function on top
  axs[2].plot(x,fx,'m',linewidth=3,label='Orig. func.')
  axs[2].set(xlabel='x',ylabel=r'y = $\int df/dx$',title='Cumulative sum of derivative')
  axs[2].legend()

  plt.tight_layout()
  plt.show()

# In [ ]

# %% [markdown]
# # Confirm with x**2

# In [ ]
# x-axis grid and function
x = np.linspace(-1,4,301)
fx = x**2

df,idf = derivAndIntegral(x,fx)
plotTheFunctions()

# In [ ]

# %% [markdown]
# # Exercise 2: Explore some other functions

# In [ ]
# x-axis grid and function
x = np.linspace(-1,4,73)
fx = x**3 + 4

# another option
x = np.linspace(-np.pi,np.pi,193)
fx = x**3/10 - np.pi*np.exp(-x**2) + np.sin(4*x)

df,idf = derivAndIntegral(x,fx)
plotTheFunctions()

# In [ ]

