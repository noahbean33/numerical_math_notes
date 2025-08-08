# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_rulesTheorems_meanValueTheorem.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation rules and theorems
# ### LECTURE: CodeChallenge: All about the mean value theorem
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
# # Exercise 1: An algorithm to solve the MVT

# In [ ]
def solveMVT(f,a,c):

  # compute the derivative
  df = sym.diff(f)

  # plug in a and c
  f_a = f.subs(x,a)
  f_c = f.subs(x,c)

  # compute the right-hand-side of the equation for b
  RHS = (f_c-f_a) / (c-a)

  # solve for b
  b = sym.solve(df-RHS)
  b = np.array(b).astype(np.float64)

  # return elements of b that are between a and c
  return b[np.bitwise_and(a<b,b<c)]

# In [ ]
# test using the function from the MVT video
x = sym.symbols('x')

expr = 2*x**2 - 3*x + 1
a = -1
c = 2

solveMVT(expr,a,c)

# In [ ]

# %% [markdown]
# # Exercise 2: Create an informative plot

# In [ ]
def makeThePlot():
  
  # solve the mean-value-theorem problem
  b = solveMVT(expr,a,c)


  # convert to callable function
  funLambda = sym.lambdify(x,expr)
  funLambda_df = sym.lambdify(x,sym.diff(expr))

  # get the function values
  xx = np.linspace(a-2,c+2,400)
  yy = funLambda(xx)


  # set up the figure
  plt.figure(figsize=(8,6))

  # plot the function
  plt.plot(xx,yy,label='f(x)')

  # plot vertical lines for a & c
  plt.plot([a,a],[np.min(yy),np.max(yy)],'r:',label='a')
  plt.plot([c,c],[np.min(yy),np.max(yy)],'m:',label='c')

  # plot dots for a & c
  plt.plot(a,funLambda(a),'ro')
  plt.plot(c,funLambda(c),'mo')

  # plot secant line
  plt.plot([a,c],[funLambda(a),funLambda(c)],'g:',label='secant')

  # plot tangent line(s)
  for i,bb in enumerate(b):
    plt.plot([bb,bb],[np.min(yy),np.max(yy)],'--',label=f'$b_{i}$',color=(.7,.7,.7))
    plt.plot(bb,funLambda(bb),'ks')
    tangX = [bb-1,bb+1]
    tangY = funLambda_df(bb)*(np.array(tangX)-bb) + funLambda(bb)
    plt.plot(tangX,tangY,'k--',label=f'tangent$_{i}$')

  # make the plot look a bit nicer
  plt.xlim(xx[[0,-1]])
  plt.legend(bbox_to_anchor=(1,1))
  plt.xlabel('x')
  plt.ylabel('f(x)')
  plt.title(f'$f(x) = {sym.latex(expr)}$')
  plt.show()

# In [ ]
makeThePlot()

# In [ ]

# %% [markdown]
# # Exercise 3: Explore different functions

# In [ ]
# now using some different functions
expr = 2*x**3 - 3*x**2 + 1
a,c = -1,2

makeThePlot()

# In [ ]
# a sine...
expr = sym.sin(x)
a,c = 0,1.8*np.pi

makeThePlot()

# In [ ]

# %% [markdown]
# # Exercise 4: Difficult functions

# In [ ]
f = 2*x / (2*x**2+1)**2
a,c = -2,1

# quick plot
sym.plot(f)

# doesn't work...
# makeThePlot()

# why not? -> let's see what we're trying to solve
f_a = f.subs(x,a)
f_c = f.subs(x,c)
RHS = (f_c-f_a) / (c-a)

# solve for b
sols = sym.solve(sym.diff(f)-RHS)

# let's see if any are real-valued
for s in sols:
  print(sym.N(s))

# In [ ]
# "simple" trig function
f = sym.cos(x**2)
a,c = -np.pi/2,6*np.pi/7

# quick plot
sym.plot(f,(x,a,c))

# doesn't work...
# makeThePlot()

# why not? -> let's see what we're trying to solve
f_a = f.subs(x,a)
f_c = f.subs(x,c)
RHS = (f_c-f_a) / (c-a)

# the expression to solve for
sym.diff(f)-RHS

# In [ ]
sym.solve(sym.diff(f)-RHS)

# In [ ]

# %% [markdown]
# # Exercise 5: Approximate a solution in numpy

# In [ ]
# lambdify the function
mvt_fun = sym.lambdify(x,sym.diff(f)-RHS)

# high-res approximation
xx = np.linspace(a,c,10001)
gridSearch = mvt_fun(xx)

# visualize the approximation
plt.plot(xx[[0,-1]],[0,0],'--',label='Solutions touch this line',color='gray')
plt.plot(xx,gridSearch,label='grid search')
plt.plot(xx,np.abs(gridSearch),label='abs(grid search)')

plt.legend()
plt.show()

# In [ ]
# find points close to zero
from scipy.signal import find_peaks
b = xx[find_peaks(-np.abs(gridSearch))[0]]
b

# In [ ]
# convert to function
funLambda = sym.lambdify(x,f)
funLambda_df = sym.lambdify(x,sym.diff(f))

# get the function values
xx = np.linspace(a-2,c+2,400)
yy = funLambda(xx)


# set up the figure
plt.figure(figsize=(8,6))

# plot the function
plt.plot(xx,yy,label='f(x)')

# plot vertical lines for a & c
plt.plot([a,a],[np.min(yy),np.max(yy)],'r:',label='a')
plt.plot([c,c],[np.min(yy),np.max(yy)],'m:',label='c')

# plot dots for a & c
plt.plot(a,funLambda(a),'ro')
plt.plot(c,funLambda(c),'mo')

# plot secant line
plt.plot([a,c],[funLambda(a),funLambda(c)],'g:',label='secant')

for i,bb in enumerate(b):
  plt.plot([bb,bb],[np.min(yy),np.max(yy)],'--',label=f'$b_{i}$',color=(.7,.7,.7))
  plt.plot(bb,funLambda(bb),'ks')
  tangX = [bb-1,bb+1]
  tangY = funLambda_df(bb)*(np.array(tangX)-bb) + funLambda(bb)
  plt.plot(tangX,tangY,'k--',label=f'tangent$_{i}$')

# make the plot look a bit nicer
plt.xlim(xx[[0,-1]])
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'$f(x) = {sym.latex(f)}$')
plt.show()

# In [ ]

