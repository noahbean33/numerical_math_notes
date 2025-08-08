# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_functions_polynomials.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Functions
# ### LECTURE: CodeChallenge: Polynomials
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
# # Exercise 1: Random polynomials in numpy

# In [ ]
xx = np.linspace(-5,5,100)

# random coefficients
coefs = np.random.randn(4)

# construct a random polynomial and a title
fname = '$y = '
y = np.zeros(len(xx))
for i,c in enumerate(coefs):
  y += c*xx**i
  fname += '+ '[int(c<0)] + f'{c:.2f}x^{i} '


# and plot
plt.plot(xx,y,linewidth=2)
plt.ylim([-30,30])
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.title(fname + '$')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Polynomials in sympy

# In [ ]
# "import" symbolic variable x
from sympy.abc import x

# define fraction parts
top = x**2 - 2*x
bot = x**2 - 4

# and function
sy = top / bot

sym.plot(sy,(x,-3,3),xlim=[-3,3],ylim=[-10,10],title=f'$y = {sym.latex(sy)}$');

# In [ ]

# %% [markdown]
# # Exercise 3: Estimate a sine wave with polynomials

# In [ ]
# order
order = 10

# initialize
x = np.linspace(-2*np.pi,2*np.pi,100)
z = np.zeros(len(x))

# loop over polynomial orders
for n in range(1,order+1):

  # polynomial for this order
  thisfun = (-1)**(n+1) * ( (x**(2*n-1))/np.math.factorial(2*n-1) )

  # plot this piece
  plt.plot(x,thisfun,'--',linewidth=.8)

  # and sum
  z += thisfun

# plot the sum
plt.plot(x,z,'k',linewidth=3,label=f'Sum over {order} terms')
plt.plot(x[::5],np.sin(x[::5]),'bo',markerfacecolor='w',linewidth=3,label='sin(x)')
plt.ylim([-5,5])
plt.xlim(x[[0,-1]])
plt.legend()
plt.show()

# In [ ]

# In [ ]

