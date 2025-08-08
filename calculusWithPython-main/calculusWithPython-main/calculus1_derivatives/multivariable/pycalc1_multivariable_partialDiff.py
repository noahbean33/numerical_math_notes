# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_multivariable_partialDiff.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Multivariable differentiation
# ### LECTURE: CodeChallenge: Partial differentiation
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from IPython.display import display,Math

# better image resolution
import matplotlib_inline.backend_inline as disPlay
disPlay.set_matplotlib_formats('svg')

# In [ ]

# %% [markdown]
# # Exercise 1: Differentiate a multivariable function

# In [ ]
x,y = sym.symbols('x,y')

# create the function
fxy = x**2/sym.pi**3 + sym.sin(y)
display(Math('f(x,y) = %s' %sym.latex(fxy)))

# In [ ]
# partial derivatives
df_x = sym.diff(fxy,x)
display(Math('\\frac{\partial f}{\partial x} = %s' %sym.latex(df_x)))
print(' ')

df_y = sym.diff(fxy,y)
display(Math('\\frac{\partial f}{\partial y} = %s' %sym.latex(df_y)))

# In [ ]

# %% [markdown]
# # Exercise 2: Visualize the function and its partial derivatives

# In [ ]
# lambdify expressions
fxy_lam = sym.lambdify((x,y),fxy)
dfx_lam = sym.lambdify((x,y),df_x)
dfy_lam = sym.lambdify((x,y),df_y)


# and evaluate at specific points
xx  = np.linspace(0,2*np.pi,21)
X,Y = np.meshgrid(xx,xx)

# In [ ]
# now draw them
_,axs = plt.subplots(1,3,figsize=(8,5))
axs[0].imshow(fxy_lam(X,Y),extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='upper')
axs[0].set_title(f'$f(x,y) = {sym.latex(fxy)}$')

axs[1].imshow(dfx_lam(X,Y),extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='upper')
axs[1].set_title(f'$f_x = {sym.latex(df_x)}$')

axs[2].imshow(dfy_lam(X,Y),extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='upper')
axs[2].set_title(f'$f_y = {sym.latex(df_y)}$')

for a in axs:
  a.set_xlabel('x')
  a.set_ylabel('y')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Now with a cross-term

# In [ ]
# the function
fxy = x**2 + sym.sin(y) + x**2*sym.sin(y)
display(Math('f(x,y) = %s' %sym.latex(fxy)))


# partial derivatives
df_x = sym.diff(fxy,x)
print(' ')
display(Math('\\frac{\partial f}{\partial x} = %s' %sym.latex(df_x)))
print(' ')

df_y = sym.diff(fxy,y)
display(Math('\\frac{\partial f}{\partial y} = %s' %sym.latex(df_y)))

# In [ ]
# lambdify expressions
fxy_lam = sym.lambdify((x,y),fxy)
dfx_lam = sym.lambdify((x,y),df_x)
dfy_lam = sym.lambdify((x,y),df_y)


# now draw them
_,axs = plt.subplots(1,3,figsize=(8,5))
axs[0].imshow(fxy_lam(X,Y),extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='upper')
axs[0].set_title(f'$f(x,y) = {sym.latex(fxy)}$')

axs[1].imshow(dfx_lam(X,Y),extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='upper')
axs[1].set_title(f'$f_x = {sym.latex(df_x)}$')

axs[2].imshow(dfy_lam(X,Y),extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='upper')
axs[2].set_title(f'$f_y = {sym.latex(df_y)}$')

for a in axs:
  a.set_xlabel('x')
  a.set_ylabel('y')

plt.tight_layout()
plt.show()

# In [ ]

