# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_integrate_sympyIntegrate.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integrating functions
# ### LECTURE: More on sympy integration
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
# # Indefinite integration in sympy

# In [ ]
# symbolic variable
x = sym.symbols('x')

# In [ ]
# the function to integrate
fx = 4*x

# its integral
int_fx = sym.integrate(fx,x)

# not the integral! Just useful for display.
int_fx_display = sym.Integral(fx)

# show the results
display(Math('\int %s \, dx = %s + C' %(sym.latex(fx),sym.latex(int_fx))))
display(Math('%s = %s + C' %(sym.latex(int_fx_display),sym.latex(int_fx))))

# In [ ]

# %% [markdown]
# # Definite integrals

# In [ ]
# the function to integrate
fx = x**2 + 4*x

# integration limits
a,b = -2,1

# its definite integral
defint_fx = sym.integrate(fx,(x,a,b))

# not the integral! Just useful for display.
defint_fx_display = sym.Integral(fx,(x,a,b))

# show the results
display(Math('\int_{%s}^{%s} (%s) \, dx = %s' %(a,b,sym.latex(fx),sym.latex(defint_fx))))
print('')
display(Math('%s = %s' %(sym.latex(defint_fx_display),sym.latex(defint_fx))))

# In [ ]

# %% [markdown]
# # Definite integration via FTC-2

# In [ ]
# indefinite integral
int_fx = sym.integrate(fx,x)

# subtraction method (FTC-2)
int_fx.subs(x,b) - int_fx.subs(x,a)

# In [ ]

# %% [markdown]
# # Visualizing f(x) and F(x)

# In [ ]
# lambdify the function and its integral
fx_l = sym.lambdify(x,fx)
Fx_l = sym.lambdify(x,int_fx)

# x-axis grids for plotting the functions and integrated area
xx = np.linspace(a-1,b+1,137)
xi = np.linspace(a,b,123)

# and plot!
plt.figure(figsize=(10,6))

# patch corresponding to area
plt.fill_between(xi,fx_l(xi),alpha=.3,label='Area of interest')

# function lines
plt.plot(xx,fx_l(xx),label=r'f(x) = $%s$' %sym.latex(fx))
plt.plot(xx,Fx_l(xx),label=r'$\int f(x) dx=%s$' %sym.latex(int_fx))

# vertical lines for integration bounds
plt.axvline(a,linestyle='--',color=[1,.7,.7])
plt.axvline(b,linestyle='--',color=[.7,.7,1])
plt.axhline(0,linestyle=':',color=[.8,.8,.8],zorder=-5)

# dots on F(x) at bounds
plt.plot(a,Fx_l(a),'ro',label=r'$F(%g) = %.2f$' %(a,Fx_l(a)))
plt.plot(b,Fx_l(b),'bo',label=r'$F(%g) = %.2f$' %(b,Fx_l(b)))

# final niceties
plt.xlabel('x')
plt.ylabel(r'f(x) or $\int f(x)dx$')
plt.xlim(xx[[0,-1]])
plt.legend(fontsize=14)
plt.show()

# In [ ]

# In [ ]
# Reminder to self ;) repeat code with constant function

# In [ ]

