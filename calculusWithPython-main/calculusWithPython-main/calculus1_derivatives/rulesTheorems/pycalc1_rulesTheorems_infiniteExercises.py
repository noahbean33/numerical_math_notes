# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_rulesTheorems_infiniteExercises.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation rules and theorems
# ### LECTURE: CodeChallenge: More differentiation exercises
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
# # Exercise 1: Practice the product and quotient rules

# In [ ]
# create variable x
x = sym.symbols('x')

# random polynomial coefficient and order
randCoefs = np.random.choice((-5,-4,-3,-2,-1,1,2,3,4,5),2)

# list of functions to sample from
funList = [ sym.cos(x),sym.sin(x),sym.log(x),sym.exp(x),randCoefs[0]*x**randCoefs[1] ]

# create the practice problem by selecting two terms to multiply
funs = np.random.choice(funList,2,replace=False)
fx = funs[0]*funs[1]

# In [ ]
print(f'Differentiate this!\n')
fx

# In [ ]
# testing d.args
d = sym.calculus.util.continuous_domain(x,x,sym.S.Reals) 
len(d.args)

# In [ ]
def plotTheFun(fx):

  # "default" domain
  D = [-3*np.pi,3*np.pi]

  # possibly change start of domain
  domain = sym.calculus.util.continuous_domain(fx,x,sym.S.Reals)
  if len(domain.args)==2:
    if domain.args[0].start==0: # note the addition of .args to access the first element in the set!
      D[0] = .001
  elif domain.start==0:
    D[0] = .001

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
plotTheFun(fx)

# In [ ]
# print out the sympy-computed derivative
print(f"Here's the derivative!\n")
sym.diff(fx)

# In [ ]

# %% [markdown]
# # Exercise 2: Product and chain rules

# In [ ]
# create variable x
x = sym.symbols('x')

# random polynomial coefficient and order
randCoefs = np.random.choice((1,2,3,4,5),2)

# list of functions to sample from
funList = [ sym.cos(x),sym.sin(x),sym.log(x),sym.exp(x),randCoefs[0]*x**randCoefs[1] ]

# create the practice problem by selecting two terms to multiply
funs = np.random.choice(funList,3,replace=False)
fx = funs[0]*funs[1].subs(x,funs[2])

# In [ ]
print(f'Differentiate this!\n')
fx

# In [ ]
plotTheFun(fx)

# In [ ]
# print out the sympy-computed derivative
print(f"Here's the derivative!\n")
sym.diff(fx)

# In [ ]

# %% [markdown]
# # Exercise 3: Implicit differentiation

# In [ ]
# create variables x and y
x,y = sym.symbols('x,y')

# random polynomial coefficient and order
randCoefsX = np.random.choice((-5,-4,-3,-2,-1,1,2,3,4,5),2)
randCoefsY = np.random.choice((-5,-4,-3,-2,-1,1,2,3,4,5),2)

# list of functions to sample from
funListX = [ sym.cos(x),sym.sin(x),sym.log(x),sym.exp(x),randCoefsX[0]*x**randCoefsX[1] ]
funListY = [ sym.cos(y),sym.sin(y),sym.log(y),sym.exp(y),randCoefsY[0]*y**randCoefsY[1] ]

# create the practice problem by selecting two terms to multiply
funsX = np.random.choice(funListX,2,replace=False)
funsY = np.random.choice(funListY,2,replace=False)
fx = funsX[0]*funsY[0] + funsX[1]*funsY[1]

# In [ ]
print(f'Differentiate this!\n')
fx

# In [ ]
# now plot
sym.plot_implicit(fx,(x,-2,3),(y,-4,2));
sym.plot_implicit(sym.idiff(fx,y,x))

# In [ ]
# implicit differentiation (input order is expression, y, x)
sym.idiff(fx,y,x)

# In [ ]

