# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_differentiation_linearity.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation fundamentals
# ### LECTURE: CodeChallenge: Linearity of differentiation
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from IPython.display import Math,display

# In [ ]

# %% [markdown]
# # Exercise 1: Derivative of summed terms

# In [ ]
# two functions and two scalars
x,a,b,c = sym.symbols('x,a,b,c')

# one function with summed terms
fun = a*x**2 + b*x**3 + c*sym.exp(2*x)

# derivative
display(Math('\\text{Derivative of combined function: }%s' %sym.latex(sym.diff(fun,x))))
print('\n')

combined = ''
for piece in fun.args:
  display(Math('\\text{Derivative of } %s \\text{ is } %s' 
               %(sym.latex(piece),sym.latex(sym.diff(piece,x)))))
  combined += sym.latex(sym.diff(piece,x)) + ' + '

print('\n')
display(Math('\\text{Combination of individual components: }%s' %combined[:-2]))

# In [ ]

# %% [markdown]
# # Exercise 2: Scalar multiples

# In [ ]
print("f'[ax**2] :")
print(sym.diff( a*x**2 ,x))

print(' ')
print("a * f'[x**2] :")
print(a*sym.diff( x**2 ,x))

# In [ ]

# %% [markdown]
# # Exercise 3: Full linearity test

# In [ ]
display(Math("\\frac{d}{dx}(ax^2 + b\\cos(x)) = %s" 
             %sym.latex(sym.diff( a*x**2 + b*sym.cos(x) ,x))))

print('\n')

display(Math("a\\frac{d}{dx}(x^2) + b\\frac{d}{dx}(\\cos(x)) = %s" 
             %sym.latex(a*sym.diff(x**2,x) + b*sym.diff(sym.cos(x),x) ) ))

# In [ ]

