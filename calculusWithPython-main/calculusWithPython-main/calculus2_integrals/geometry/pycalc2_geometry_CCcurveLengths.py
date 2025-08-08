# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geometry_CCcurveLengths.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in geometry
# ### LECTURE: CodeChallenge: approximating curve lengths
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import scipy.integrate as spi
import matplotlib.pyplot as plt
from IPython.display import Math

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Exercise 1: Implement in sympy

# In [ ]
# symbolic functions
t = sym.symbols('t')

x = sym.sin(t) * (sym.exp(sym.cos(t)) - 2*sym.cos(4*t) - sym.sin(t/12)**5)
y = sym.cos(t) * (sym.exp(sym.cos(t)) - 2*sym.cos(4*t) - sym.sin(t/12)**5)

# and their derivatives
dx = sym.diff(x,t)
dy = sym.diff(y,t)

# let's see what it looks like!
display(Math('x(t) = %s' %sym.latex(x))), print('')
display(Math("x'(t) = %s" %sym.latex(dx))), print('')
print('')

display(Math('y(t) = %s' %sym.latex(y))), print('')
display(Math("y'(t) = %s" %sym.latex(dy)))

# In [ ]
# bounds
a = 0
b = 2*sym.pi

# sympy plot
sym.plot_parametric(x,y,(t,a,b));

# In [ ]

# %% [markdown]
# # Exercise 2: Curve length using spi.quad

# In [ ]
# the integrand
integrand = sym.lambdify(t,sym.sqrt(dx**2 + dy**2) )

# and now to integrate
length_quad,_ = spi.quad(integrand,a,b)

print(f'Curve length using spi.quad: {length_quad}')

# In [ ]

# %% [markdown]
# # Exercise 3: Lambdify and visualize

# In [ ]
x_l = sym.lambdify(t,x)
y_l = sym.lambdify(t,y)

dx_l = sym.lambdify(t,dx)
dy_l = sym.lambdify(t,dy)

tt = np.linspace(float(a),float(b),999)

_,axs = plt.subplots(2,2,figsize=(10,7))

axs[0,0].plot(x_l(tt),y_l(tt),'k')
axs[0,0].set(xlabel='$x$',ylabel='$y$',title='Parametric functions')

axs[0,1].plot(tt,x_l(tt),label='x(t)')
axs[0,1].plot(tt,y_l(tt),label='y(t)')
axs[0,1].set(xlim=tt[[0,-1]],xlabel='$t$',ylabel='$x$ or $y$',title='Functions by t')

axs[1,0].plot(dx_l(tt),dy_l(tt),'k')
axs[1,0].set(xlabel='$dx$',ylabel='$dy$',title='Their derivatives')

axs[1,1].plot(tt,dx_l(tt),label="x'(t)")
axs[1,1].plot(tt,dy_l(tt),label="y'(t)")
axs[1,1].set(xlim=tt[[0,-1]],xlabel='$t$',ylabel='$dx$ or $dy$',title='Derivatives by t')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Curve length approximations as a function of N

# In [ ]
# sample counts
Ns = np.linspace(10,1000,40).astype(int)

# initialize
lengths_npy  = np.zeros(len(Ns))
lengths_spi = np.zeros(len(Ns))

# create a figure
_,axs = plt.subplots(1,2,figsize=(12,5))

# loop over the values of \Delta t
for idx,n in enumerate(Ns):

  ttt = np.linspace(float(a),float(b),n)
  dt = ttt[1] - ttt[0]

  # all-numpy solution
  dxSquared = (dx_l(ttt)*dt)**2
  dySquared = (dy_l(ttt)*dt)**2
  lengths_npy[idx] = np.sum( np.sqrt( dxSquared + dySquared ) )

  # Simpson's rule
  integrand = np.sqrt((dx_l(ttt))**2 + (dy_l(ttt))**2)
  lengths_spi[idx] = spi.simpson(integrand,dx=dt)

  # plot the first and last curves
  if idx==0:
    axs[0].plot(x_l(ttt),y_l(ttt),'.-',label='N = %g'%n)
  elif idx==(len(Ns)-1):
    axs[0].plot(x_l(ttt),y_l(ttt),'.-',label='N = %g'%n)



# more plotting
axs[0].set(xlabel='$x$',ylabel='$y$',title='Example curves for two Ns')
axs[0].legend()

axs[1].plot(Ns,lengths_npy,'s-',label='Numpy')
axs[1].plot(Ns,lengths_spi,'x-',label="Simpson's")
axs[1].axhline(length_quad,color='k',linestyle='--',label='spi.quad')
axs[1].set(xlabel='N',ylabel='Curve length',title='Curve length by N')
axs[1].legend()

plt.tight_layout()
plt.show()

# In [ ]

