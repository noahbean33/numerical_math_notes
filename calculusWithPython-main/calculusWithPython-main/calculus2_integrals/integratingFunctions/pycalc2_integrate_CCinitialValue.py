# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_integrate_CCinitialValue.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integrating functions
# ### LECTURE: CodeChallenge: Code the initial value problem
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

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

# In [ ]

# %% [markdown]
# # Exercise 1: Implement the initial value problem

# In [ ]
# symbolic variables
x,C = sym.symbols('x,C')

# problem givens and constraint
df = 2*x + 3
initial_vals = [1,2] # first number is x_0, second number is f(x_0)

# for exercise 2
# df = sym.cos(x) + x**sym.Rational(5,2)
# df = sym.cos(x) * (x**sym.Rational(5,2))
# initial_vals = [1,sym.pi]


# step 1: integrate to find f(x)
fx = sym.integrate(df,x) + C

# step 2: solve for C
constant = sym.solve( fx.subs(x,initial_vals[0]) - initial_vals[1] ,C)[0]

# print the results!
display(Math('\\text{PROBLEM:}'))
display(Math("\\text{Given } f'(x) = %s \\text{, and } f(%g)=%g" %(sym.latex(df),initial_vals[0],initial_vals[1])))
print(' ')
display(Math('\\text{SOLUTION:}'))
display(Math("f(x) = " + sym.latex(fx.subs(C,constant))))

# In [ ]

# %% [markdown]
# # Visualization

# In [ ]
# x-axis points to evaluate functions
xx = np.linspace(-2,2,41)

# a set of incorrect coefficients, to show that other solutions are parallel curves
wrongConstants = np.linspace(float(constant)-2,float(constant)+2,12)

# loop over the incorrect coefficients
for ci in wrongConstants:

  # get function values
  y = [fx.subs(C,ci).subs(x,i) for i in xx]

  # plot them
  plt.plot(xx,y,color=[.6,.6,.6],linewidth=.4)


# now plot the real value
y = [fx.subs(C,constant).subs(x,i) for i in xx]
plt.plot(xx,y,color=[.5,0,.6],linewidth=5)

plt.xlim(xx[[0,-1]])
plt.show()

# In [ ]

# In [ ]
# a brief aside on why to use sym.Rational
x**(5/3)
x**sym.Rational(5,3)

