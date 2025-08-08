# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_limits_confirmTrigLimits.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Limits
# ### LECTURE: CodeChallenge: Confirm the trig limits
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
# # Exercise 1: The important ones

# In [ ]
# functions in sympy
phi = sym.symbols('phi')
fx1 = sym.sin(phi) / phi
fx2 = (sym.cos(phi)-1) / phi


# print out limits
print('\nLimit of sin(x)/x as x approaches zero:')
print( sym.limit(fx1,phi,0,dir='+-') )

print('\nLimit of (cos(x)-1)/x as x approaches zero:')
print( sym.limit(fx2,phi,0,dir='+-') )

# In [ ]

