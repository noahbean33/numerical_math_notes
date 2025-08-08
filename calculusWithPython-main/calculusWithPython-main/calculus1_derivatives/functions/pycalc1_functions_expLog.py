# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_functions_expLog.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Functions
# ### LECTURE: CodeChallenge: exp and log
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
# # Exercise 1: estimate e

# In [ ]
# estimate e

n = [ 1, 2, 5, 10 ]

for i in n:

  # estimate e using this value of n
  e = (1+(1/i))**i

  # print it out
  print(f'n: {i:2.0f},  est.e: {e:6.5f},  diff to e: {np.exp(1)-e:8.7f}')

# In [ ]

# %% [markdown]
# # Exercise 2: visualize e's approach

# In [ ]
# define vector of n
n = np.arange(1,1001)

# define differences between estimation and "true" value
eDiffs = np.exp(1) - (1+1/n)**n

plt.plot(n,eDiffs)
plt.xlabel('n')
plt.ylabel('Difference to np.exp(1)')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Exploring e in numpy

# In [ ]

# domain for x
xDomain = [ -2,2 ]

numSteps = 41
x = np.linspace(xDomain[0],xDomain[1],numSteps)

# functions
y1 = np.exp(x)
y2 = np.exp(x**2)
y3 = np.exp((-x)**2)
y4 = np.exp(-(x**2))
y5 = np.exp(x)**2


# and plot
plt.figure(figsize=(8,6))
plt.plot(x,y1,linewidth=2,label='$y=e^x$')
plt.plot(x,y2,linewidth=2,label='$y=e^{x^2}$')
plt.plot(x,y3,'--',linewidth=2,label='$y=e^{(-x)^2}$')
plt.plot(x,y4,linewidth=2,label='$y=exp(-(x^2))$')
plt.plot(x,y5,linewidth=2,label='$y=exp(x)^2$')
plt.legend()
plt.grid()
plt.xlim([x[0],x[-1]])
plt.ylim([-1,10])
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Exploring e in sympy

# In [ ]
### in sympy

# create a symbolic variable
s_beta = sym.var('beta')

# define the function
s_y = sym.exp(s_beta) - sym.log(s_beta) - np.exp(1)

# use sympy's plotting engine
sym.plot(s_y,(s_beta,xDomain[0],xDomain[1]),
         title=f'$f(\\beta) = {sym.latex(s_y)}$',
         xlabel='x',ylabel='$y=f(\\beta)$')

plt.show()

# In [ ]

# %% [markdown]
# # Exercise 5: exp and log

# In [ ]
x = np.linspace(.001,4,30)
plt.plot(x,np.log(x),linewidth=2,label='log(x)')

x = np.linspace(-4,4,30)
plt.plot(x,np.exp(x),linewidth=2,label='exp(x)')

plt.plot(x,np.log( np.exp(x) ),label='log(exp(x))')
plt.plot(x,np.exp( np.log(x) ),'o',label='exp(log(x))')

plt.axis('square')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# In [ ]

