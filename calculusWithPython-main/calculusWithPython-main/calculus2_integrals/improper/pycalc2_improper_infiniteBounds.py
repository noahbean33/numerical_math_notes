# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_improper_infiniteBounds.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Improper integrals
# ### LECTURE: Two infinite bounds
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

# In [ ]
# define the function
x = sym.symbols('x')
fx = 4*x / (x**4+2)

# sympy plot
sym.plot(fx,(x,-10,10));

# In [ ]
# print the indefinite integral
display(Math('%s = %s' %(sym.latex(sym.Integral(fx)),sym.latex(sym.integrate(fx)))))

# In [ ]
# definite integrals from each side, and total
display(Math('%s = %s' %(sym.latex(sym.Integral(fx,(x,-sym.oo,0))),sym.latex(sym.integrate(fx,(x,-sym.oo,0))))))
print('')
display(Math('%s = %s \\approx %s' %(sym.latex(sym.Integral(fx,(x,0,sym.oo))),sym.latex(sym.integrate(fx,(x,0,sym.oo))),
                                     sym.latex(sym.integrate(fx,(x,0,sym.oo)).evalf()))))
print('')
display(Math('%s = %s' %(sym.latex(sym.Integral(fx,(x,-sym.oo,sym.oo))),sym.latex(sym.integrate(fx,(x,-sym.oo,sym.oo))))))

