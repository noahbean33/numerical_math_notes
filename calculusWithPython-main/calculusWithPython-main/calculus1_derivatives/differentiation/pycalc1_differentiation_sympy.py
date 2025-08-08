# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_differentiation_sympy.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation fundamentals
# ### LECTURE: CodeChallenge: derivatives in sympy
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
# # Exercise 1: Derivative of a line

# In [ ]
# create sympy expression object and plot
x = sym.symbols('x')
fx = (5/4)*x + 9/4
sym.plot(fx,(x,-1,6),axis_center=[0,0]);

# In [ ]
# compute the derivative
sym.diff(fx,x)

# In [ ]
# store as a variable
dydx = sym.diff(fx,x)
print(f'The derivative is {dydx}')

# In [ ]

# %% [markdown]
# # Exercise 2: Derivative of a polynomial

# In [ ]
# create sympy expression object and plot
x = sym.symbols('x')
fx = x**2
sym.plot(fx,(x,-1,6));

# In [ ]
# store as a variable
dydx = sym.diff(fx,x)
dydx

# In [ ]

# %% [markdown]
# # Exercise 3: Plot and query the derivative

# In [ ]
sym.plot(dydx,(x,-1,6));

# In [ ]
# query the derivative at a particular point
somePoints = [-1,0,2]

for p in somePoints:
  print(f'The derivative at x={p} is dy/dx={dydx.subs(x,p)}')

# In [ ]

