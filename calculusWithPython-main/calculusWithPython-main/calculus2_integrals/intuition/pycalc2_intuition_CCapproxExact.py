# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_intuition_CCapproxExact.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration and applications
# ## SECTION: Intuition for integration
# ### LECTURE: CodeChallenge: Approximate exact integrals
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202506

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import Math

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# %% [markdown]
# # Exercise 1: Algebraic approximations of indefinite integrals

# In [ ]
# a python function to approximate an integral using computational methods
def numericalIntegral(x,fx):

  # find the x-axis coordinate of x=0
  zeroIdx = np.argmin(abs(x-0))

  # cumulative sum (discrete integral)
  dx = x[1] - x[0]
  idf = np.cumsum(fx) * dx
  idf -= idf[zeroIdx] # normalize so that idf(0)=0
  idf += fx[zeroIdx]  # then add constant from original function

  return idf

# In [ ]
# create a symbolic function and gets its analytical integral
from sympy.abc import x

# the function and its integral
fx_s = sym.sin(x) * x**3
intf_s = sym.integrate(fx_s,x)

# print the function and its antiderivative
display(Math('$f(x) = %s$' %sym.latex(fx_s)))
display(Math('$F(x) = %s$' %sym.latex(intf_s)))

# lambidfy both functions
fx_l = sym.lambdify(x,fx_s)
intf_l = sym.lambdify(x,intf_s)

# In [ ]
# get its computational (empirical) integral
xx = np.linspace(-np.pi,np.pi,31)
intf_c = numericalIntegral(xx,fx_l(xx))

# In [ ]
# plot for comparison
plt.plot(xx,intf_c,'ms',label='Numerical (numpy)')
plt.plot(xx,intf_l(xx),'k',label='Analytical (sympy)')

plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\int f(x) dx$')
plt.xlim(xx[[0,-1]])
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Approximation error for different $\Delta$x

# In [ ]
# compute RMS for different [\Delta x]'s
deltaXs = np.logspace(np.log10(.5),np.log10(.001),20)
RMSs = np.zeros(len(deltaXs))

for i,dx in enumerate(deltaXs):

  # compute empirical integral
  xx = np.arange(-np.pi,np.pi+dx,dx)
  intf_a = numericalIntegral(xx,fx_l(xx))

  # compute approximation error as root-mean-squared error
  RMSs[i] = np.mean((intf_l(xx) - intf_a)**2)**(1/2)



# visualize the error
_,ax = plt.subplots(1,figsize=(10,6))
ax.plot(deltaXs,RMSs,'ks-',linewidth=2,markerfacecolor='w',markersize=10)
ax.invert_xaxis()
ax.axhline(0,linestyle='--',color='m')

ax.set(xlabel=r'$\Delta x$',ylabel='Approximation error (a.u.)')
ax.set_xscale('log')
# ax.set_yscale('log')
ax.legend(['Approximations','True integral'])
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Geometric approximation of definite integrals

# In [ ]
# function bounds
bounds = [-1,2]

# create a definite integral object for display
expr = sym.Integral(fx_s,(x,bounds)) # doesn't actually integrate

# compute the analytic definite integral
defIntegral = sym.integrate(fx_s,(x,bounds[0],bounds[1])) # note list unpacking in the previous line

# print the symbol and numerical result
Math('%s \;=\; %s \;\\approx\; %s' %(sym.latex(expr),sym.latex(defIntegral),sym.N(defIntegral)))

# In [ ]
# resolutions
areas = np.zeros(len(deltaXs))

# loop over resolutions
for i,dx in enumerate(deltaXs):
  xx = np.arange(bounds[0],bounds[1]+dx,dx)

  # compute and store area (using list comprehension here)
  areas[i] = np.sum([ fx_l(xi)*dx for xi in xx ])

# In [ ]
# create an axis object (easier to invert the axis...)
_,ax = plt.subplots(1,figsize=(10,5))

# plot the results
ax.plot(deltaXs,areas,'ks-',linewidth=2,markerfacecolor='w',markersize=10,label='Empirical estimate')
ax.axhline(defIntegral, linestyle='--',color='m',label='True integral')

ax.legend()
ax.invert_xaxis()
ax.set_xscale('log')
ax.set(xlabel=r'$\Delta x$',ylabel='Area (estimate of definite integral)')
plt.show()

# In [ ]

