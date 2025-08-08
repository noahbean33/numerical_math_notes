# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_limits_discontinuities.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Limits
# ### LECTURE: CodeChallenge: Limits at discontinuities
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
# # Exercise 1: Jump discontinuity in numpy

# In [ ]
# piecewise function

def fx(x):
  x = np.array(x)
  y = np.zeros(x.shape)

  tol = 10e-4
  y[x<-tol] = np.sin(x[x<-tol]*np.pi)
  y[x>tol] = -(x[x>tol]-2)**2
  y[np.abs(x)<tol] = 1.5
  return y


xx = np.linspace(-1,2,10001)
plt.plot(xx,fx(xx),'o')
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.title('A function with a jump discontinuity')
plt.show()

# In [ ]
# limit as x approaches 0 from the left

# x-axis coordinates getting closer to 0
xFromLeft  = -np.logspace(np.log10(.1),np.log10(.002),10)
xFromRight =  np.logspace(np.log10(.002),np.log10(.1),10)

# function values
limitFromLeft  = fx(xFromLeft)
limitFromRight = fx(xFromRight)

print(f'Limit approaches {limitFromLeft[-1]} from the left.')
print(f'Limit approaches {limitFromRight[0]} from the right.')
print(f'Function value at x=0:  {fx(0)}.')


# plot
plt.plot(xFromLeft,limitFromLeft,'s',label='From the left')
plt.plot(xFromRight,limitFromRight,'o',label='From the right')
plt.plot(0,fx(0),'rx',label='y=f(0)')
plt.legend()
plt.title(f'Function value at x=0 is {fx(0)}.')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Jump discontinuity in sympy

# In [ ]
# "import" symbolic variable x
from sympy.abc import x

# list function pieces
piece1 = sym.sin(x*sym.pi)
piece2 = 1.5
piece3 = -(x-2)**2

# put them together with conditions
fx = sym.Piecewise( 
      (piece1,x<0),
      (piece2,sym.Eq(x,0)),
      (piece3,x>0) 
      )


# plot
sym.plot(fx,(x,xx[0],xx[-1]))
plt.show()
fx

# In [ ]
# test limits
print('Limit as x approaches 0 from the left:')
print( sym.N(sym.limit(fx,x,0,dir='-')) )

print('\nLimit as x approaches 0 from the right:')
print( sym.limit(fx,x,0,dir='+') )

print('\nTwo-sided limit as x approaches 0:')
print( sym.limit(fx,x,0,dir='+-') )

print('\n\nFunction value at limit:')
print( sym.N(sym.limit(fx,x,0)) )

# In [ ]

# %% [markdown]
# # Exercise 3: Infinite discontinuity

# In [ ]
fx = 3/(1-x**2)

sym.plot(fx,(x,-2,2),ylim=[-10,10])
plt.show()

# In [ ]
# test limits
print('Limit as x approaches -1 from the left:')
print( sym.N(sym.limit(fx,x,-1,dir='-')) )

print('\nLimit as x approaches -1 from the right:')
print( sym.limit(fx,x,-1,dir='+') )

print('\nTwo-sided limit as x approaches -1:')
print( sym.limit(fx,x,-1,dir='+-') )

# In [ ]

# %% [markdown]
# # Exercise 4: Oscillating discontinuity

# In [ ]
fx = sym.sin(1/x)
sym.plot(fx,(x,-1,1))
plt.show()

# In [ ]
# test limits
print('Limit as x approaches 0 from the left:')
print( sym.N(sym.limit(fx,x,0,dir='-')) )

print('\nLimit as x approaches 0 from the right:')
print( sym.limit(fx,x,0,dir='+') )

print('\nTwo-sided limit as x approaches 0:')
print( sym.limit(fx,x,0,dir='+-') )

# In [ ]

