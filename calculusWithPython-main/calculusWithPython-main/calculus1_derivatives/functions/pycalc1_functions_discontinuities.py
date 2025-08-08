# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_functions_discontinuities.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Functions
# ### LECTURE: CodeChallenge: Discontinuities
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

resolution = .1
xx = np.arange(-1,2,resolution)

# list function definitions
pieces    = [0]*3 # initialize list
pieces[0] = np.sin(xx*np.pi)
pieces[1] = 1.5*np.ones(len(xx))
pieces[2] = -(xx-2)**2

# and their x-axis value domains
xdomains    = [0]*3 # initialize list
xdomains[0] = xx<0
xdomains[1] = np.abs(xx)<resolution/2
xdomains[2] = xx>0



# and plot
marker = '-o-'
for i in range(len(pieces)):
  plt.plot(xx[xdomains[i]],pieces[i][xdomains[i]],marker[i],linewidth=2)

plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.title('A function with a jump discontinuity')
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
      (piece2,sym.Eq(x,0)), # note: not x==0!
      (piece3,x>0) 
      )


# use sympy's plotting engine
sym.plot(fx,(x,xx[0],xx[-1]))

plt.show()

# In [ ]
# f(0)=1.5 not shown in the plot, but sympy knows it:
fx.subs(x,0)

# In [ ]

# %% [markdown]
# # Exercise 3: Removable discontinuity

# In [ ]
fx = np.sin(xx*np.pi) + xx**2
fx[np.argmin(np.abs(xx-0))] = np.pi

plt.plot(xx,fx,'o')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Infinite discontinuity

# In [ ]
xx = np.linspace(-2,2,1001)
fx = 3/(1-xx**2)

plt.plot(xx,fx,'k',linewidth=3)
plt.plot([-1,-1],[-20,20],'--',color=[.6,.6,.6])
plt.plot([1,1],  [-20,20],'--',color=[.6,.6,.6])
plt.ylim([-20,20])
plt.xlim([xx[0],xx[-1]])
plt.show()

# In [ ]
# identify continuous domain in sympy
sfx = 3/(1-x**2)
sym.calculus.util.continuous_domain(sfx,x,sym.Interval(-2,2))

# In [ ]
# identify singularities in sympy
sym.singularities(sfx,x)

# In [ ]

# %% [markdown]
# # Exercise 5: Oscillating discontinuity

# In [ ]
# in numpy
xx = np.linspace(-1,2,101)
fx = np.sin(1/(xx-1))

plt.plot(xx,fx)
plt.show()

# In [ ]
# in sympy
sfx = sym.sin(1/(x-1))
sym.plot(sfx,(x,-1,2));

# In [ ]

