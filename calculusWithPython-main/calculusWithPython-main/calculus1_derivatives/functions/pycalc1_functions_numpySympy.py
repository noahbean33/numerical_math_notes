# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_functions_numpySympy.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Functions
# ### LECTURE: CodeChallenge: Math in Python
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# better image resolution
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# In [ ]

# %% [markdown]
# # Exercise 1: function in numpy

# In [ ]
### in numpy

# domain for x
xDomain = [ -2,2 ]

# grid resolution ("step" parameter)
resolution = .1

# create grid of x-axis values
x = np.arange(xDomain[0],xDomain[1]+resolution,resolution)

# Or:
numSteps = 41
x = np.linspace(xDomain[0],xDomain[1],numSteps)

# function
y = x**2 + 3*x**3 - x**4


# and plot
plt.figure(figsize=(8,6))
plt.plot(x,y,linewidth=2,label='$y=x^2 + 3x^3 - x^4$')
plt.legend()
plt.grid()
plt.xlim([x[0],x[-1]])
plt.ylim([np.min(y),np.max(y)])
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: in sympy

# In [ ]
# create a symbolic variable
s_beta = sym.var('beta')

# define the function
s_y = s_beta**2 + 3*s_beta**3 - s_beta**4

# use sympy's plotting engine
sym.plot(s_y,(s_beta,xDomain[0],xDomain[1]),axis_center='auto',
         title=f'$f(\\beta) = {sym.latex(s_y)}$',
         xlabel='x',ylabel=None)#'$y=f(\\beta)$')

plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: convert sympy to numpy

# In [ ]
##

# create the function object
fx = sym.lambdify(s_beta,s_y)

# evaluate one x-axis value
fx(2)

# In [ ]
# get a list of values in numpy and plot with matplotlib

yy = fx(x)


# and plot
plt.figure(figsize=(8,6))
plt.plot(x,yy,linewidth=2,label='$y=x^2 + 3x^3 - x^4$')
plt.legend()
plt.grid()
plt.xlim([x[0],x[-1]])
plt.ylim([np.min(yy),np.max(yy)])
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.show()

# In [ ]

