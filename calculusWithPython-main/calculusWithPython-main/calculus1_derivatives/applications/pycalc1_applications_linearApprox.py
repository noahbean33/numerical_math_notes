# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_applications_linearApprox.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Applications
# ### LECTURE: CodeChallenge: Linear approximations
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
# # Exercise 1: Approximate a point

# In [ ]
x = sym.symbols('x')

# define the function in sympy
fx = sym.sqrt(x) + 2*x

# compute its derivative
df = sym.diff(fx,x)


# function that takes f, df, x0, a, and returns estimate at a
def linearApproximation(f,df,x0,a):
  #  y  =  m     x    +  b
  return df.subs(x,a)*(x0-a) + f.subs(x,a)

# In [ ]
x0 = 4.37 # value to estimate
a = 4 # convenient point

linearApproxValue = linearApproximation(fx,df,x0,a)
trueValue = fx.subs(x,x0)

print(f'Linear approximation: {linearApproxValue}')
print(f'Sympy calculation   : {trueValue}')

# In [ ]

# %% [markdown]
# # Exercise 2: Different starting values

# In [ ]
# list of starting values
startVals = np.linspace(5,14,49)
x0 = np.mean(startVals)
trueValueX0 = fx.subs(x,x0)

# initializations
linapprox = np.zeros(len(startVals))
approxErrors = np.zeros(len(startVals))

# do the approximations and store the results
for i,a in enumerate(startVals):
  linapprox[i] = linearApproximation(fx,df,x0,a)
  approxErrors[i] = trueValueX0 - linapprox[i]


plt.plot(startVals,approxErrors,'s-')
plt.xlabel('Initial guess')
plt.ylabel('Error from "true" value')
plt.show()

# In [ ]
# lambdify the function for plotting
xx = np.linspace(0,14,50)
fx_fun = sym.lambdify(x,fx)

# plot the results!
plt.plot(startVals,linapprox,'ro',label='approximations')
plt.plot(xx,fx_fun(xx),label='f(x)')
plt.plot(np.mean(startVals),trueValueX0,'ks',label='f(x$_0$)')

plt.xlabel('x'), plt.ylabel('y=f(x)')
plt.xlim(xx[[0,-1]])
plt.legend()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: A different function

# In [ ]
# redefine variable to limit to real field
# x = sym.symbols('x',real=True)

# define the function in sympy
fx = sym.Abs( sym.sin(x) )

# compute its derivative
df = sym.diff(fx,x)

# In [ ]
x0 = 5*np.pi/8 # value to estimate
a = np.pi/2 # convenient point

linearApproximation(fx,df,x0,a) 

# In [ ]
# list of starting values
startVals = np.linspace(-np.pi,2*np.pi,99)
x0 = np.mean(startVals)
trueValueX0 = fx.subs(x,x0)

# initializations
linapprox = np.zeros(len(startVals))
approxErrors = np.zeros(len(startVals))

# do the approximations and store the results
for i,a in enumerate(startVals):
  linapprox[i] = linearApproximation(fx,df,x0,a) # after error, set real=True when defining x
  approxErrors[i] = linapprox[i] - fx.subs(x,x0)


plt.plot(startVals,approxErrors,'s-')
plt.xlabel('Initial guess')
plt.ylabel('Error from "true" value')
plt.show()

# In [ ]
# lambdify the function for plotting
xx = np.linspace(-np.pi,2*np.pi,99)
fx_fun = sym.lambdify(x,fx)

# plot the results!
plt.plot(startVals,linapprox,'ro',label='approximations')
plt.plot(xx,fx_fun(xx),label='f(x)')
plt.plot(np.mean(startVals),trueValueX0,'ks',label='f(x$_0$)')

plt.xlabel('x'), plt.ylabel('y=f(x)')
plt.xlim(xx[[0,-1]])
plt.legend()
plt.show()

# In [ ]

