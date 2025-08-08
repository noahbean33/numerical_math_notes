# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_rulesTheorems_cx.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation rules and theorems
# ### LECTURE: CodeChallenge: Derivative of c^x and x^x
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# better image resolution
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

from IPython.display import display,Math

# In [ ]

# %% [markdown]
# # Exercise 1: Differentiating 2^x

# In [ ]
# create a symbolic variable
x = sym.symbols('x')

# define expression
fx = 2**x

# differentiate it!
df = sym.diff(fx,x)

# print out both
display(Math(f'f(x) = {sym.latex(fx)}'))
display(Math(f"f'(x) = {sym.latex(df)}"))

# In [ ]
# lambdify
fx_lam = sym.lambdify(x,fx)
df_lam = sym.lambdify(x,df)

# create an x-grid
xx = np.linspace(-2,3,1001)

# plot!
plt.plot(xx,fx_lam(xx),label=f'$f(x) = {sym.latex(fx)}$',linewidth=2)
plt.plot(xx,df_lam(xx),label=f"$f'(x) = {sym.latex(df)}$",linewidth=2)
plt.plot(xx,np.exp(xx),label=f"$f'(x) = e^x$",linewidth=2)

plt.legend()
plt.xlabel('x')
plt.ylabel("f(x) or f'(x)")
plt.xlim(xx[[0,-1]])
plt.grid()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Differentiating x^x

# In [ ]
# define expression
fx = x**x

# differentiate it!
df = sym.diff(fx,x)

# print out both
display(Math(f'f(x) = {sym.latex(fx)}'))
display(Math(f"f'(x) = {sym.latex(df)}"))

# In [ ]
# lambdify
fx_lam = sym.lambdify(x,fx)
df_lam = sym.lambdify(x,df)

# create an x-grid
xx = np.linspace(-1,2,1001)

# plot!
plt.plot(xx,np.real(fx_lam(xx)),label=f'$f(x) = {sym.latex(fx)}$',linewidth=2)
plt.plot(xx,df_lam(xx),label=f"$f'(x) = {sym.latex(df)}$",linewidth=2)

plt.legend()
plt.xlabel('x')
plt.ylabel("f(x) or f'(x)")
plt.xlim(xx[[0,-1]])
plt.grid()
plt.show()

# In [ ]

# In [ ]
# just a bit of fun with (-c)**x

k = np.linspace(-1,-5,100)
q = np.linspace(-np.pi/2,np.pi/2,1000)

for i,ki in enumerate(k):
  z = np.power(ki,q,dtype=complex)
  plt.plot(np.real(z),np.imag(z),linewidth=.7,color=[.8*i/len(k),.7-.6*i/len(k),i/len(k)])

plt.axis('off')
plt.show()

# In [ ]

