# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_rulesTheorems_implicit.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation rules and theorems
# ### LECTURE: CodeChallenge: implicit differentiation
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [1]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# better image resolution
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# In [ ]

# %% [markdown]
# # Exercise 1: Implicit differentiation in sympy

# In [ ]
# create two symbolic variables
x,y = sym.symbols('x,y')

# define expression (implicitly set to 0!)
expr = x*y-1

# implicit differentiation (input order is expression, y, x)
sym.idiff(expr,y,x)

# In [ ]
# now plot
sym.plot_implicit(expr,(x,-2,3),(y,-4,2));
sym.plot_implicit(sym.idiff(expr,y,x))

# In [ ]
solve4y = sym.solve(expr,y)
df = sym.idiff(expr,y,x).subs(y,solve4y[0])
sym.plot(df,xlim=(-3,3),ylim=(-10,1));

# In [ ]

# %% [markdown]
# # Exercise 2: Evaluate expression at a value

# In [ ]
# expression
expr = x**3 * y**2 - 5*x**4

# derivative via implicit differentiation
sym.idiff(expr,y,x)

# In [ ]
# substitute a symbolic variable for y
num2evalY = sym.sqrt(5)
sym.idiff(expr,y,x).subs(y,num2evalY)

# In [ ]
# substitute a numerical value for y
num2evalY = np.sqrt(5)
sym.idiff(expr,y,x).subs(y,num2evalY)

# In [ ]
# substitute symbolic variables for x and y
num2evalX = sym.sqrt(5)
num2evalY = sym.pi
sym.idiff(expr,y,x).subs([ (x,num2evalX),(y,num2evalY) ])

# In [ ]
# substitute numerical variables for x and y
num2evalX = np.sqrt(5)
num2evalY = np.pi
sym.idiff(expr,y,x).subs([ (x,num2evalX),(y,num2evalY) ])

# In [ ]

# %% [markdown]
# # Exercise 3: Derivative without a function

# In [ ]
# expression and its derivative
expr = sym.exp(x**2+y**2) - x - y
sym.idiff(expr,y,x)

# In [ ]
# plots
sym.plot_implicit(expr);
sym.plot_implicit(sym.idiff(expr,y,x),(x,0,.5),(y,-3,3));

# In [ ]
# try to solve for y
sym.solve(expr,y)

# In [ ]
# try to solve for y
sym.solve(sym.idiff(expr,y,x),y)

# In [ ]

