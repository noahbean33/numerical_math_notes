# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_functions_compositeInverse.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Functions
# ### LECTURE: CodeChallenge: Composite and inverse functions
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
# # Exercise 1a: Composite functions in numpy

# In [ ]
# define the math functions as python functions
def fx(x):
  return 2*x**2 - 4

def gx(x):
  return 7*np.abs(x) + 3

# In [ ]
xx = np.linspace(-5,5,200)

# evaluate composite functions
fgx = fx(gx(xx))
gfx = gx(fx(xx))

# and plot
plt.plot(xx,fx(xx),linewidth=2,label='f(x)')
plt.plot(xx,gx(xx),linewidth=2,label='g(x)')
plt.plot(xx,fgx,linewidth=2,label='f(g(x))')
plt.plot(xx,gfx,'.',linewidth=2,label='g(f(x))')

plt.plot(xx[[0,-1]],[0,0],'k--',linewidth=.3)
plt.plot([0,0],[-10,50],'k--',linewidth=.3)

plt.ylim([-10,50])
plt.xlim(xx[[0,-1]])
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.legend()
plt.show()

# %% [markdown]
# # Exercise 1b: Composite functions in numpy

# In [ ]
def fx(x):
  return np.sin(x)
def gx(x):
  return np.log(x)
def hx(x):
  return 2*x**2 + 5

xx = np.linspace(-100*np.pi,100*np.pi,1001)

# evaluate composite function (try different orders!)
f3 = fx(gx(hx(xx)))

# and plot
plt.plot(xx,f3,linewidth=2,label='f(g(h(x)))')
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.legend()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Inverse function in numpy

# In [ ]
def fx(x):
  return np.log(2*x)

def gx(x):
  return np.exp(x)/2

xx = np.linspace(.01,5,50)

# evaluate composite functions
fgx = fx(gx(xx))
gfx = gx(fx(xx))

# and plot
plt.plot(xx,fx(xx),linewidth=2,label='f(x)')
plt.plot(xx,gx(xx),linewidth=2,label='g(x)')
plt.plot(xx,fgx,linewidth=2,label='f(g(x))')
plt.plot(xx,gfx,'.',linewidth=2,label='g(f(x))')

plt.ylim([-5,10])
plt.xlim(xx[[0,-1]])
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.legend()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Composite functions in sympy

# In [ ]
sx = sym.symbols('x')
fx = sym.sin(sx)
gx = sym.log(sx)
hx = 2*sx**2 + 5

# evaluate composite function (try different orders!)
fun3 = fx.subs(sx,gx.subs({'x':hx}))

# and plot
sym.plot(fun3,(sx,-100,100),title=f'$y = {sym.latex(fun3)}$');

# In [ ]

# %% [markdown]
# # Exercise 4: Inverting functions in sympy

# In [ ]
# symbolic variable y
y = sym.symbols('y')

# define function
fun = 2*sx + 3
# fun = 2*sx + sym.sin(sx)

# invert
sym.solve(y-fun,sx)[0]

# In [ ]
# show that the composition of the function and its inverse gives the original x
invfun = sym.solve(y-fun,sx)[0]

fun.subs(sx,invfun.subs(y,4))

# In [ ]
# and it works the other way around (watch the variable names!)
invfun.subs(y,fun.subs(sx,4))

# In [ ]

