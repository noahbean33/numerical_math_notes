# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_integrate_scipy.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integrating functions
# ### LECTURE: Numerical integration in scipy
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import Math


# NEW!
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
# # Integrate based on a function with limits

# In [ ]
# create the symbolic function in sympy
x = sym.symbols('x')
fx_sym = x**2 + 10*sym.sin(x)

# limits of integration
a = 0
b = sym.pi

# exact integration from sympy
int_sympy = sym.integrate(fx_sym,(x,a,b))

# numerical integration using scipy by directly creating a lambda expression
int_scipy, error = spi.quad(lambda t: t**2 + 10*np.sin(t), a,b)

## same result as above but converting from sympy
# fx_lam = sym.lambdify(x,fx_sym,'scipy')
# int_scipy, error = spi.quad(fx_lam, a,b)


# print the results
display(Math('\\text{Exact integral: } %s' %sym.latex(int_sympy)))
display(Math('\\text{Numerical integral from sympy: } %.8f' %sym.N(int_sympy)))
display(Math('\\text{Numerical integral from scipy: } %.8f' %int_scipy))

# In [ ]

# %% [markdown]
# # Example use-case

# In [ ]
# the function and its plot
expr = x**3 / (sym.exp(x)-1)
sym.plot(expr,(x,-1,3));

# In [ ]
# bounds
a,b = -1,3

# Attempt to integrate symbolically
int_sympy = sym.integrate(expr,(x,a,b))
int_sympy#.evalf()

# In [ ]
# using scipy

# same result as above but converting from sympy
fx_lam = sym.lambdify(x,expr,'scipy')
int_scipy, error = spi.quad(fx_lam, a,b)

# print the results
display(Math('\\text{Exact integral: } %s' %sym.latex(int_sympy)))
# display(Math('\\text{Numerical integral from sympy: } %.8f' %sym.N(int_sympy)))
display(Math('\\text{Numerical integral from scipy: } %.8f' %int_scipy))

# In [ ]

# %% [markdown]
# # Integrate based on empirical data

# In [ ]
# some random data
N = 500 # number of data points (try increasing!)
xx = np.linspace(0,4*np.pi,N)
dx = xx[1]-xx[0]

# the data
y = np.cos(xx) + np.linspace(0,2,N) + np.random.randn(N)

# cumulative sum scaled by dx
antideriv = np.cumsum(y) * dx

# discrete integral using scipy (trapezoidal rule)
int_trap = spi.trapezoid(y,x=xx,dx=dx)

# visualize it
_,axs = plt.subplots(1,2,figsize=(12,5))

axs[0].plot(xx,y,'ko',markerfacecolor='w',markersize=8)
axs[0].set(xlabel='x',label='Data value',xlim=[xx[0]-.2,xx[-1]+.2])
axs[0].set_title(f'The data (N = {N})')

axs[1].plot(xx,antideriv,'ko-',markerfacecolor='w',markersize=8)
axs[1].axhline(antideriv[-1],color=[.7,.7,.7],linestyle='--',zorder=-3)
axs[1].text(xx[-1]*.5,antideriv[-1]*.97,f'Max val = {antideriv[-1]:.4f}',color=[.7,.7,.7],va='top',ha='center')
axs[1].set(xlabel='x',label='Integrated data value',xlim=[xx[0]-.2,xx[-1]+.2])
axs[1].set_title(f'Antiderivative (definite integral = {int_trap:.4f})')

plt.tight_layout()
plt.show()

# In [ ]

