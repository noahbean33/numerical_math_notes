# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_improper_dicontinuities.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Improper integrals
# ### LECTURE: Functions with discontinuities
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import display,Math

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Jump discontinuities

# In [ ]
x = sym.symbols('x')

fx = sym.Piecewise((x/2,x<0),(x**2+1,x>=0))
a,b = -2,1

sym.plot(fx,(x,a-1,b+1))
sym.integrate(fx,(x,a,b))

# In [ ]

# %% [markdown]
# # Infinite discontinuity

# In [ ]
fx = 1 / sym.sqrt(x-1)
a,b = 1,2

sym.plot(fx,(x,a-1,b+1),ylim=[0,10])
sym.integrate(fx,(x,a,b))

# In [ ]

# %% [markdown]
# # Removable discontinuity

# In [ ]
fx = (x**2-2*x) / (x**2-4)
a,b = 0,4

sym.plot(fx,(x,a-1,b+1),ylim=[-1,1])
sym.integrate(fx,(x,a,b)).evalf()

# In [ ]

