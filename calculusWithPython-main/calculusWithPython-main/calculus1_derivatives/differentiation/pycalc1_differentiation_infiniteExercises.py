# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_differentiation_infiniteExercises.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation fundamentals
# ### LECTURE: CodeChallenge: Infinite derivatives exercises
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

# In [ ]

# %% [markdown]
# # Exercise 1: Generate a random function

# In [ ]
# create variable x
x = sym.symbols('x')

# random polynomial order
polyOrder = np.random.randint(1,5)

# list of other functions to include
funList = [ sym.cos(x),sym.sin(x),sym.log(x),sym.exp(x) ]

# initialize the function
fx = 0

# add the polynomial terms
for i in range(polyOrder):
  fx += np.random.randint(-5,6)*x**i

# add the transcendental functions
for f in np.random.choice(funList,2,replace=False):
  fx += np.random.choice((-1,1))*f

# In [ ]
print(f'Differentiate this!\n')
fx

# In [ ]

# %% [markdown]
# # Exercise 2: Lambdify and plot

# In [ ]
# "default" domain
D = [-3,3]

# possibly change start of domain
domain = sym.calculus.util.continuous_domain(fx,x,sym.S.Reals)
if domain.start==0:
  D[0] = .001

# In [ ]
# convert expression to function
fxfun = sym.lambdify(x,fx)

# compute function values and empirical derivative
xx = np.linspace(D[0],D[1],1234)
y  = fxfun(xx)
dy = np.diff(y) / np.mean(np.diff(xx))


# plot the function
_,axs = plt.subplots(2,1,figsize=(6,6))
axs[0].plot(xx,y,linewidth=2)
axs[0].set_title(f'$f(x)={sym.latex(fx)}$')

# and the derivative
axs[1].plot(xx[:-1],dy,linewidth=2)
axs[1].set_title('df')


# prettify both axes
for a in axs:
  a.set_xlim(xx[[0,-1]])
  a.set_xlabel('x')
  a.set_ylabel('y')


plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Derivative in sympy

# In [ ]
print(f"Here's the derivative!\n")
sym.diff(fx)

# In [ ]

