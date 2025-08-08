# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geoApprox_calculart.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Geometric approximations
# ### LECTURE: Calculart! Riemannesque animations
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


# NEW: import animation module and set defaults
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='jshtml')


# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Calculart 1: Random Riemann bars

# In [ ]
# function for the function
def fx(u):
  return np.cos(2*np.pi*u) * np.exp(-u**2)

# In [ ]
# define boundaries and random deltax
a = -2
b = 2
deltax = np.random.rand()/20


# setup the plot
_,axs = plt.subplots(1,figsize=(10,6))

# initialize at the lower bound
y = fx(a)
bp = a + deltax

# keep going until the upper bound
while bp<b:

  # compute the function value and draw rectangle
  y = fx(bp)
  faCo = np.random.rand(1) * np.array([.7,.3,.9])
  axs.fill_between([bp,bp+deltax],[y,y],edgecolor=None,facecolor=faCo)

  # update deltax and next breakpoint
  deltax = np.random.rand()/20
  bp += deltax


plt.axis('off')
plt.show()

# In [ ]

# %% [markdown]
# # Calculart 2: Random Riemann: The movie

# In [ ]
# define boundaries and random deltax
a = -2
b = 2
deltax = np.random.rand()/20


# initialize at the lower bound
y = fx(a)
bp = a + deltax

data2animate = []
# keep going until the upper bound
while bp<b:

  # compute the function value and draw rectangle
  y = fx(bp)
  faCo = np.random.rand(1) * np.array([.7,.3,.9])

  data2animate.append([y,bp,deltax,faCo])

  # update deltax and next breakpoint
  deltax = np.random.rand()/20
  bp += deltax

data2animate[6]

# In [ ]
def aframe(idx):
  axs.clear()
  axs.set(ylim=[-1,1],xlim=[a,b])
  axs.axis('off')

  # draw all bars up to the current index
  for i in range(idx+1):
    y,bp,deltax,faCo = data2animate[barorder[i]]
    axs.fill_between([bp,bp+deltax],[y,y],edgecolor=None,facecolor=faCo)


# Setup figure
fig, axs = plt.subplots(1,figsize=(10,6))
barorder = np.random.permutation(len(data2animate))
ani = animation.FuncAnimation(fig, aframe, frames=len(data2animate), interval=30, repeat=True);
ani

# In [ ]
# save as a gif
writergif = animation.PillowWriter(fps=30)
ani.save('RiemannesqueAnimation.gif', writer=writergif)

# In [ ]

# %% [markdown]
# # Calculart 3: Riemann zeta function

# In [ ]
# library with numerically stable zeta function
import mpmath

# initialize
n = 18
yy = np.linspace(0,np.pi**3,1201)
z = np.zeros(len(yy),dtype=complex)

# setup the figure with background color
plt.figure(figsize=(6,6),facecolor='k')

# compute the values
for i,y in enumerate(yy):
  s = 1j*y + 1/2
  # z[i] = np.sum( 1/(np.arange(1,n+1)**s) )
  z[i] = mpmath.zeta(s)

# finish the plot
plt.scatter(np.real(z),np.imag(z),50,c=np.abs(z),marker='^',cmap='RdPu_r')
plt.axis('equal')
plt.axis('off')
plt.show()

# In [ ]

# %% [markdown]
# # Calculart 4: The zeta movie

# In [ ]
# function to draw the plots
def aframe(maxY):

  # compute the values
  yy = np.arange(0,maxY,.1)
  z = np.zeros(len(yy),dtype=complex)
  for i,y in enumerate(yy):
    z[i] = mpmath.zeta(1j*y + 1/2)

  # update the black function line
  plth1.set_xdata(np.real(z))
  plth1.set_ydata(np.imag(z))

  # update the pink pointer
  plth2.set_xdata([0,np.real(z[-1])])
  plth2.set_ydata([0,np.imag(z[-1])])

  return plth1,plth2


# setup figure
fig,ax = plt.subplots(1,figsize=(6,6))
plth1, = ax.plot(0,0,'k',linewidth=2)
plth2, = ax.plot(0,0,'m',linewidth=.5)
ax.set(xlim=[-2,4],ylim=[-3,3])
ax.axis('off')

# run the animation
maxY = np.linspace(.1,np.pi**3,100)
ani = animation.FuncAnimation(fig, aframe, np.concatenate((maxY, maxY[::-1])), interval=50, repeat=True)
ani

# In [ ]
# save as a gif
writergif = animation.PillowWriter(fps=30)
ani.save('RiemannZetaAnimation.gif', writer=writergif)

# In [ ]

