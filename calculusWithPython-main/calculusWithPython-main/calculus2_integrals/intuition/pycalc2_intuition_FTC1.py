# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_intuition_FTC1.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration and applications
# ## SECTION: Intuition for integration
# ### LECTURE: The fundamental theorem of calculus, Part 1
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

# In [ ]

# %% [markdown]
# # FTC1: the integral of a derivative is the function

# In [ ]
# create symbolic variables
x,C = sym.symbols('x,C')

# create a function
fx = (x-1)**2

# take its derivative
df = sym.diff(fx,x)

# integrate the derivative
id = sym.integrate(df,x) + C

# print the results
display(Math('\\text{Original function: } f(x) = %s' %sym.latex(fx)))
display(Math("\\text{Function derivative: } f'(x) = %s" %sym.latex(df)))
display(Math("\\text{Integral of derivative: } F(x) = %s" %sym.latex(sym.factor(id))))

# In [ ]
# show that F(f(x)')==f(x) with the constant
display(Math('\\text{Original function: } f(x) = %s' %sym.latex(sym.expand(fx))))
display(Math("\\text{Integral of derivative: } F(x) = %s" %sym.latex(id)))

# In [ ]

# %% [markdown]
# # Visualize the functions

# In [ ]
# quick-n-dirty plotting
sym.plot(fx)
sym.plot(id.subs(C,1));

# In [ ]
# plotting with matplotlib

# lambdify all functions
fx_l = sym.lambdify(x,fx)
df_l = sym.lambdify(x,df)
id_l = sym.lambdify(x,id.subs(C,0)) # experiment with C!

# define x-axis grid
xx = np.linspace(-1,4,23)

# and plot
plt.figure(figsize=(10,7))
plt.plot(xx,fx_l(xx),'k',label='f(x)')
plt.plot(xx,df_l(xx),'rd',label="f'(x)")
plt.plot(xx,id_l(xx),'bo',markersize=8,label="F(f'(x))")

plt.gca().set(xlim=xx[[0,-1]],xlabel='x',ylabel='y or dy/dx')
plt.legend()
plt.grid()
plt.show()

# In [ ]

