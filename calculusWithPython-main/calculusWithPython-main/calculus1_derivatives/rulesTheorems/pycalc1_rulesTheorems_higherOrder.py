# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_rulesTheorems_higherOrder.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation rules and theorems
# ### LECTURE: CodeChallenge: Derivatives of derivatives...
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
from IPython.display import display

# In [ ]

# %% [markdown]
# # Exercise 1: Third derivatives in numpy

# In [ ]
# define the x-axis grid
dx = .01
x = np.arange(-np.pi,np.pi,dx)

# define the function
f = .1*x**4 + np.exp(-x**2) + np.cos(x)

# compute the derivatives
df = np.diff(f) / dx
ddf = np.diff(df) / dx
dddf = np.diff(ddf) / dx

# In [ ]
# plot!

# setup the figure and define the titles
_,axs = plt.subplots(4,1,figsize=(4,10))
titles = [ "f(x)","f'(x)","f''(x)","$f^{(3)}(x)$" ]

# plot the lines
axs[0].plot(x,f)
axs[1].plot(x[:-1],df)
axs[2].plot(x[:-2],ddf)
axs[3].plot(x[:-3],dddf)


# applies to all axes
for a,t in zip(axs,titles):
  a.set_xlim(x[[0,-1]])
  a.set_title(t)
  a.spines['right'].set_visible(False)
  a.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Using np.diff second input

# In [ ]
# specify the derivative order as the second input to np.diff
dfI = np.diff(f,1) / dx
ddfI = np.diff(f,2) / dx**2
dddfI = np.diff(f,3) / dx**3

# test whether they're the same
print(f'Difference for first derivative:  {np.mean(np.abs(dfI-df))}')
print(f'Difference for second derivative: {np.mean(np.abs(ddfI-ddf))}')
print(f'Difference for third derivative:  {np.mean(np.abs(dddfI-dddf))}')

# In [ ]

# %% [markdown]
# # Exercise 3: Higher-order derivatives in sympy

# In [ ]
# define the function
x = sym.symbols('x')
sf = 3*x**4 + sym.exp(-x**2) + sym.cos(x)


print(f'The function is:')
display(sf), print()

# compute and show the derivatives
for i in range(1,4):
  print(f'The {i}-order derivative is:')
  display(sym.diff(sf,x,i)) # try sym.simplify!
  print()

# In [ ]

