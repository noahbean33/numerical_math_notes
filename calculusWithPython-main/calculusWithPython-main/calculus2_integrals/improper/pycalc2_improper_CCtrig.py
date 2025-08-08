# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_improper_CCtrig.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Improper integrals
# ### LECTURE: CodeChallenge: Improper trig integrals
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import display,Math
import scipy.integrate as spi

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Exercise 1: Improper trig integrals

# In [ ]
x = sym.symbols('x')

# the function
fx = sym.cos(x)
# fx = sym.cos(x**2) # uncomment for exercise 2

# lambdify the function (twice to get net/total area)
fx_l_net = sym.lambdify(x,fx)
fx_l_tot = sym.lambdify(x,sym.Abs(fx))

# define the upper integration limits
upperLims = np.linspace(np.pi/4,8*np.pi,63)

# initialize the results vectors
defintNet = np.zeros(len(upperLims))
defintTot = np.zeros(len(upperLims))

# run the calculations!
for i,U in enumerate(upperLims):
  defintNet[i],_ = spi.quad(fx_l_net,0,U)
  defintTot[i],_ = spi.quad(fx_l_tot,0,U)

# In [ ]
# plot the function
_,axs = plt.subplots(1,2,figsize=(14,4))
xx = np.linspace(0,upperLims[-1],1001)
axs[0].plot(xx,fx_l_net(xx),linewidth=2)
axs[0].axhline(0,linestyle='--',color=[.7,.7,.7],zorder=-3)
axs[0].set(xlabel='Angle (x)',ylabel=r'$y = %s$'%sym.latex(fx),
           xlim=[0,upperLims[-1]],title='Function')

# and its integral as a function of upper bounds
axs[1].plot(upperLims,defintNet,label='Net area',linewidth=2)
axs[1].plot(upperLims,defintTot,label='Total area',linewidth=2)
axs[1].axhline(0,linestyle='--',color=[.7,.7,.7],zorder=-3)
axs[1].set(xlabel='Upper limit',ylabel='Area',xlim=upperLims[[0,-1]],
           title='Areas')
axs[1].legend()

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Analytic integrals in sympy

# In [ ]
# analytic improper integrals

fx = sym.Abs(sym.cos(x))
display(Math('%s = %s' % (sym.latex(sym.Integral(fx,(x,0,sym.oo))),sym.latex(sym.integrate(fx,(x,0,sym.oo))))))
print('')

fx = sym.cos(x)
display(Math('%s = %s' % (sym.latex(sym.Integral(fx,(x,0,sym.oo))),sym.latex(sym.integrate(fx,(x,0,sym.oo))))))
print('')

fx = sym.cos(x**2)
display(Math('%s = %s' % (sym.latex(sym.Integral(fx,(x,0,sym.oo))),sym.latex(sym.integrate(fx,(x,0,sym.oo))))))

# In [ ]

