# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_multivariable_hopd.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Multivariable differentiation
# ### LECTURE: CodeChallenge: 2D functions in sympy
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import display,Math

# In [ ]

# %% [markdown]
# # Exercise 1: The whole shebang

# In [ ]
x,y = sym.symbols('x,y')

# create the function
fxy = sym.cos(2*x*y**2) + x**2 - y*sym.exp(x)
fxy

# In [ ]
# two first-order partial derivatives
dfx = sym.diff(fxy,x)
dfy = sym.diff(fxy,y)

display(Math('f_x = %s' %sym.latex(dfx)))
print(' ')
display(Math('f_y = %s' %sym.latex(dfy)))

# In [ ]
# two second-order same-variable partial derivatives
dfxx = sym.diff( sym.diff(fxy,x) ,x)
dfyy = sym.diff( sym.diff(fxy,y) ,y)

display(Math('f_{xx} = %s' %sym.latex(dfxx)))
print(' ')
display(Math('f_{yy} = %s' %sym.latex(dfyy)))

# In [ ]
# two mixed partial derivatives
dfxy = sym.diff( sym.diff(fxy,x) ,y)
dfyx = sym.diff( sym.diff(fxy,y) ,x)

display(Math('f_{xy} = %s' %sym.latex(dfxy)))
print(' ')
display(Math('f_{yx} = %s' %sym.latex(dfyx)))

# In [ ]

