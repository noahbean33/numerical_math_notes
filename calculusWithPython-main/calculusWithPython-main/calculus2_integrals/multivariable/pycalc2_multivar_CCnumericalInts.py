# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_multivar_CCnumericalInts.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Multivariable integration
# ### LECTURE: CodeChallenge: numerical double integrals
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

# %% [markdown]
# # Exercise 1: Integrate a function in sympy

# In [ ]
# the variables
t,s = sym.symbols('t,s')

# the function
f_ts = sym.sin(sym.sqrt(t**2 + s**2)) + t/10
s_bnd = [0,2]
t_bnd = [0,1]

# definite integral
defint = sym.integrate(f_ts,
                      (t,t_bnd[0],t_bnd[1]),
                      (s,s_bnd[0],s_bnd[1]))

# print the function and its integral
display(Math('f(t,s) = %s' %sym.latex(f_ts)))
print('')
display(Math('\\int\\int f(t,s) \,dt \,ds= %s' %sym.latex(sym.integrate(f_ts,(t,s)))))
print('')
display(Math('%s = %s' %(sym.latex(sym.Integral(f_ts,(t,t_bnd[0],t_bnd[1]),(s,s_bnd[0],s_bnd[1]))),sym.latex(defint))))

# In [ ]

# %% [markdown]
# # Exercise 2: Code and visualize the function

# In [ ]
# Define the function to integrate
def f_xy(x,y):
  return np.sin(np.sqrt(x**2 + y**2)) + x/10

# bounds of integration
a,b = -5,5

# create a meshgrid to evaluate the function
x = np.linspace(a,b,50)
y = np.linspace(a,b,50) # same as x
dx = x[1]-x[0]
X,Y = np.meshgrid(x,y)

# the function
function_landscape = f_xy(X,Y)

# and visualize
fig,ax = plt.subplots(1,figsize=(7,5))
c = ax.contourf(X,Y,function_landscape, levels=50, cmap='turbo', vmin=-1,vmax=1)
fig.colorbar(c,ax=ax)
ax.set(xlabel='x',ylabel='y')
ax.set_title(r'$f(x,y) = \sin\left(\sqrt{x^2+y^2}\right) + x/10$')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Discrete integral using numpy

# In [ ]
# initialize partial integral matrices
integral_x = np.zeros_like(X)
integral_y = np.zeros_like(Y)
integral_xy = np.zeros_like(Y)
integral_yx = np.zeros_like(Y)

# constant of integration (used for Exercise 6)
C = f_xy(-5,0) # use for Exercise 6
C = 0 # set C=0 for Exercise 3

# calculate the partial integrals by summing the function values
for xi in range(len(x)):
  for yi in range(len(y)):

    # integrate with respect to x, keeping y fixed
    integral_x[yi,xi] = np.sum(function_landscape[yi,:xi]) * dx + C

    # integrate with respect to y, keeping x fixed
    integral_y[yi,xi] = np.sum(function_landscape[:yi,xi]) * dx

    # now for the double-integrals (integrating the partial integrals)
    integral_xy[yi,xi] = np.sum(integral_x[:yi,xi]) * dx # note the dx^2!
    integral_yx[yi,xi] = np.sum(integral_y[yi,:xi]) * dx

# In [ ]
# the two double-integrals are equal (within precision error)
np.round(integral_xy-integral_yx,5)

# In [ ]
# Plotting
fig,axs = plt.subplots(2,2,figsize=(12,8))

# show the function
c1 = axs[0,0].contourf(X,Y,function_landscape, levels=50, cmap='turbo', vmin=-1,vmax=1)
fig.colorbar(c1, ax=axs[0,0])
axs[0,0].set_title('The function')

# the double integral
c2 = axs[0,1].contourf(X,Y,integral_xy, levels=50, cmap='turbo', vmin=-20,vmax=20)
fig.colorbar(c2, ax=axs[0,1])
axs[0,1].set_title(r'$\int\int f(x,y) \,dx\,dy$')


# integral wrt x
c3 = axs[1,0].contourf(X, Y, integral_x, levels=50, cmap='turbo',vmin=-5,vmax=5)
fig.colorbar(c3, ax=axs[1,0])
axs[1,0].set_title(r'$\int f(x,y) \,dx$')

# integral wrt y
c4 = axs[1,1].contourf(X, Y, integral_y, levels=50, cmap='turbo',vmin=-5,vmax=5)
fig.colorbar(c4, ax=axs[1,1])
axs[1,1].set_title(r'$\int f(x,y) \,dy$')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Integrate using scipy

# In [ ]
# Initialize matrices for the results
integral_xs = np.zeros_like(X)
integral_ys = np.zeros_like(Y)
integral_xys = np.zeros_like(X)

# integrate.quad assumes variables in "reverse" order
def f_xySwap(y,x):
  return np.sin(np.sqrt(x**2 + y**2)) + x/10


# Now integrate using scipy's integrate.quad and integrate.dblquad
for xi in range(len(x)):
  for yi in range(len(y)):

    # Integrate with respect to x, keeping y fixed
    integral_xs[yi,xi],_ = spi.quad(lambda x: f_xy(x,y[yi]), a,x[xi])

    # Repeat for y, integrating with respect to y, keeping x fixed
    integral_ys[yi,xi],_ = spi.quad(lambda y: f_xy(x[xi],y), a,y[yi])

    # And now for the double integral
    # Note: dblquad integrates first over y (inner) and then over x (outer)
    integral_xys[yi,xi],_ = spi.dblquad(f_xySwap, a, x[xi], lambda x: a, lambda x: y[yi])

# In [ ]
# Plotting
fig,axs = plt.subplots(2,2,figsize=(12,8))

# show the function
c1 = axs[0,0].contourf(X,Y,function_landscape, levels=50, cmap='turbo', vmin=-1,vmax=1)
fig.colorbar(c1, ax=axs[0,0])
axs[0,0].set_title('The function')

# the double integral
c2 = axs[0,1].contourf(X,Y,integral_xys, levels=50, cmap='turbo', vmin=-20,vmax=20)
fig.colorbar(c2, ax=axs[0,1])
axs[0,1].set_title(r'$\int\int f(x,y) \,dx\,dy$')


# integral wrt x
c3 = axs[1,0].contourf(X, Y, integral_xs, levels=50, cmap='turbo',vmin=-5,vmax=5)
fig.colorbar(c3, ax=axs[1,0])
axs[1,0].set_title(r'$\int f(x,y) \,dx$')

# integral wrt y
c4 = axs[1,1].contourf(X, Y, integral_ys, levels=50, cmap='turbo',vmin=-5,vmax=5)
fig.colorbar(c4, ax=axs[1,1])
axs[1,1].set_title(r'$\int f(x,y) \,dy$')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 5: Draw some lines

# In [ ]
plt.figure(figsize=(8,4))

# pick a row index to plot
row = 25

plt.plot(x,function_landscape[row,:],lw=2,label='Function')
plt.plot(x,integral_xs[row,:],lw=2,label='Integral wrt x (scipy)')
plt.plot(x,integral_x[row,:],'o',lw=2,label='Integral wrt x (numpy)')

plt.xlim(x[[0,-1]])
plt.xlabel('x')
plt.ylabel(r'$f(x,y)$  or  $\int f(x,y)\,dx$')
plt.title(f'Row {row} of f(x,y) and integral_x')
plt.legend()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 6: Add a constant

# In [ ]
# see comment in Exercise 3

# In [ ]

