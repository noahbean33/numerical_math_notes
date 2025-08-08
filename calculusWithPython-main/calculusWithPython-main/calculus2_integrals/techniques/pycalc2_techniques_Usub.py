# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_techniques_Usub.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integration techniques
# ### LECTURE: U-substitution
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

# In [ ]
x,C = sym.symbols('x,C')

# define the function and its antiderivative
fx = x / (x**2 + 2)
# fx = (3*x**4+5)**6 * x**3

antideriv = sym.integrate(fx)

# show the result using latex
display(Math('%s = %s+C' %(sym.latex(sym.Integral(fx)),sym.latex(antideriv))))

# In [ ]
# the actual latex code
'%s = %s+C' %(sym.latex(sym.Integral(fx)),sym.latex(antideriv))

# In [ ]
# and visualize the function
xx = np.linspace(-2,5,351) # x-axis grid
y = [fx.subs(x,xi) for xi in xx] # function
F = [antideriv.subs(x,xi) for xi in xx] # indefinite integral


# plotting
_,axs = plt.subplots(1,figsize=(10,5))

axs.plot(xx,y,linewidth=2,label=r'f(x) = $%s$'%sym.latex(fx))
axs.plot(xx,F,linewidth=2,label=r'F(x) = $%s+0$'%sym.latex(antideriv))
axs.axhline(0,linestyle='--',color=[.8,.8,.8],zorder=-3)

axs.set(xlim=xx[[0,-1]],xlabel='x',ylabel='f(x) or F(x)')
axs.legend()
plt.show()

# In [ ]

# In [ ]
# test whether my answer matches sympy's
myAns = (3*x**4+5)**7/84
sym.expand(myAns)# == antideriv

# In [ ]

