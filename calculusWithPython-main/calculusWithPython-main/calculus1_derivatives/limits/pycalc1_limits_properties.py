# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_limits_properties.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Limits
# ### LECTURE: CodeChallenge: properties of limits
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
# # Exercise 1: lim(c*f) = c*lim(f)

# In [ ]
# a function in sympy
from sympy.abc import x

# define the function and plot
fx = x**3/3 + 100*sym.sqrt(sym.Abs(x))
sym.plot(fx,(x,-10,10),ylim=[-100,500]);fx

# In [ ]
# demonstrate the constant-factor property
c = np.random.randn()
print('   lim(c*fx):')
print( sym.N(sym.limit(c*fx,x,5)) )

print(' ')
print('   c*lim(fx):')
print( sym.N(c*sym.limit(fx,x,5)) )

# In [ ]

# %% [markdown]
# # Exercise 2: lim(f+g) = lim(f) + lim(g)

# In [ ]
f = sym.log(x) + x**2
g = sym.exp(-x) + x**3

print( sym.limit(f+g,x,np.pi) )
print( sym.limit(f,x,np.pi)+sym.limit(g,x,np.pi) )

# In [ ]
# use sym.pi or np.pi?
sym.limit(f+g,x,sym.pi)

# In [ ]

# %% [markdown]
# # Exercise 3: lim(f*g) = lim(f)lim(g)

# In [ ]
# use the same functions as above
print( sym.limit( f*g ,x,np.pi) )
print( sym.limit(f,x,np.pi)*sym.limit(g,x,np.pi) )

# In [ ]
# also for powers
print( sym.limit(f**3,x,np.pi) )
print( sym.limit(f,x,np.pi)*sym.limit(f,x,np.pi)*sym.limit(f,x,np.pi) )

# In [ ]

# %% [markdown]
# # Exercise 4: lim(f/g) = lim(f)/lim(g)

# In [ ]
# use the same functions as above
print( sym.limit( f/g ,x,np.pi) )
print( sym.limit(f,x,np.pi)/sym.limit(g,x,np.pi) )

# In [ ]
# but be mindful of ?/0
h = x**3 + x**2 + x

print( sym.limit( g,x,0) )
print( sym.limit( h,x,0) )
print( sym.limit( g/h ,x,0) )

# In [ ]
# it's still a valid function
g/h

# FYI: sym.factor(g/h)

# In [ ]

