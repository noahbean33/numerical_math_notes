# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_applications_newton.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Applications
# ### LECTURE: CodeChallenge: Newt's roots
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

# %% [markdown]
# # Exercise 1: Roots of a sympy expression

# In [ ]
from sympy.abc import x

# define the function
fx = 2*x**3 - 3

# find the real root
realRoot = sym.N(sym.solve(fx,x)[0])
realRoot

# In [ ]

# %% [markdown]
# # Exercise 2: Implement Newton's method

# In [ ]
# function to implement one iteration
def newtonIter(f,d,x0):
  return x0 - f.subs(x,x0)/d.subs(x,x0)

# In [ ]
# do one iteration

# function derivative
df = sym.diff(fx)

# start value (aka x0)
start = 1
x1 = newtonIter(fx,df,start)
print(f'Estimate of root from first iteration: {x1}')

# a second iteration
x2 = newtonIter(fx,df,x1)
print(f'Estimate of root from second iteration: {x2}')

# In [ ]

# %% [markdown]
# # Exercise 3: Iterations in a loop

# In [ ]
# multiple iterations in a loop
numIters = 7

# initialize x0 and vector of all guesses
startGuess = 1
xGuess = np.zeros(numIters) + startGuess
# xGuess = np.full(numIters,startGuess,dtype=np.float64) # equivalent to previous

# loop over iterations
for i in range(1,numIters):
  xGuess[i] = newtonIter(fx,df,xGuess[i-1])

# plot the guesses
plt.plot(xGuess,'s-')
plt.plot([0,numIters-1],[realRoot,realRoot],'r--',label='True root')
plt.xlabel('Iteration')
plt.ylabel('Root approximation')
plt.legend()
plt.show()

# In [ ]
# plot the function and its root-approximations

# lambdify the function
fx_fun = sym.lambdify(x,fx)
xx = np.linspace(-1,2,301)

# plot the function and true root
plt.plot(xx[[0,-1]],[0,0],'--',color=[.7,.7,.7])
plt.plot(xx,fx_fun(xx),linewidth=2,label='f(x)')
plt.xlim(xx[[0,-1]])
plt.ylim([-10,10])
plt.plot([realRoot,realRoot],plt.gca().get_ylim(),'--',color=[1,.6,.6],label='True root')

# (optional) change red intensity for each update
redvals = np.linspace(.2,1,len(xGuess))

# plot in a loop (can do without the loop with static color)
for g,r in zip(xGuess,redvals):
  plt.plot(g,fx_fun(g),'o',markersize=8,color=[r,.1,.3])

# finalize the plot
# plt.xlim([np.min(xGuess)*.9,np.max(xGuess)*1.1])
plt.legend()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Start further away

# In [ ]
# change variable 'startGuess' to -5

# In [ ]

# %% [markdown]
# # Exercise 5: A different function

# In [ ]
# define the function and its derivative
fx = sym.cos(x) - x**2
df = sym.diff(fx)

# find the roots using sympy
sym.solve(fx,x)

# In [ ]
# plot it to see that there are roots
sym.plot(fx,(x,-2,2));

# In [ ]
# multiple iterations in a loop
numIters = 7

# initialize x0 and vector of all guesses
startGuess = 1
xGuess = np.zeros(numIters) + startGuess

# loop over iterations
for i in range(1,numIters):
  xGuess[i] = newtonIter(fx,df,xGuess[i-1])

# plot the guesses
plt.plot(xGuess,'s-')
plt.show()

# In [ ]
# plot the function and its root-approximations

# lambdify the function
fx_fun = sym.lambdify(x,fx)
xx = np.linspace(-1.5,1.5,301)

# plot the function and true root
plt.plot(xx[[0,-1]],[0,0],'--',color=[.7,.7,.7])
plt.plot(xx,fx_fun(xx),linewidth=2,label='f(x)')
plt.xlim(xx[[0,-1]])
plt.ylim([-1,.5])

# (optional) change red intensity for each update
redvals = np.linspace(.2,1,len(xGuess))

# plot in a loop (can do without the loop with static color)
for g,r in zip(xGuess,redvals):
  plt.plot(g,fx_fun(g),'o',markersize=8,color=[r,.1,.3])

# finalize the plot
# plt.xlim([np.min(xGuess)*.9,np.max(xGuess)*1.1])
plt.show()

# In [ ]

