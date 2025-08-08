# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_multivariable_2dFun.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Multivariable differentiation
# ### LECTURE: CodeChallenge: Fun with 2D functions (numpy)
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
# # Exercise 1: 2D function in numpy using for-loops

# In [ ]
dx = np.pi/6
xx = np.arange(0,2*np.pi+dx/2,dx)
Z = np.zeros((len(xx),len(xx)))

for coli,x in enumerate(xx):
  for rowi,y in enumerate(xx):

    # note the association of row/column to y/x
    Z[rowi,coli] = np.sin(x) + np.cos(y) 


# and visualize
plt.imshow(Z,extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='lower')
plt.title('$f(x,y) = \\sin(X) + \\cos(Y)$')
plt.xlabel('x (rad.)')
plt.ylabel('y (rad.)')

# specify tick locations and labels
ticklocs = np.arange(0,2*np.pi+.1,np.pi/2)
ticklabels = ['0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$']

plt.xticks(ticklocs,labels=ticklabels)
plt.yticks(ticklocs,ticklabels)
plt.title('$f(x,y) = \sin(X) + \cos(Y)$')
plt.colorbar()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Create a grid to evaluate functions

# In [ ]
xx = np.linspace(0,2*np.pi,15)
X,Y = np.meshgrid(xx,xx)

plt.plot(xx,np.zeros(len(xx)),'o')
plt.ylim([-1,10])
plt.xlabel('x grid')
plt.ylabel('Function value')
plt.show()


_,axs = plt.subplots(1,2,figsize=(8,5))
axs[0].imshow(X,extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='lower')
axs[0].set_title('X grid')

axs[1].imshow(Y,extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='lower')
axs[1].set_title('Y grid')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Create and visualize a 2D function

# In [ ]
# now create a 2D function
Z = np.sin(X) + np.cos(Y)

# and visualize
plt.imshow(Z,extent=[xx[0],xx[-1],xx[0],xx[-1]],origin='lower')
plt.title('$f(x,y) = \sin(X) + \cos(Y)$')
plt.xlabel('x (rad.)')
plt.ylabel('y (rad.)')

# specify tick locations and labels
ticklocs = np.arange(0,2*np.pi+.1,np.pi/2)
ticklabels = ['0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$']

plt.xticks(ticklocs,labels=ticklabels)
plt.yticks(ticklocs,ticklabels)
plt.title('$f(x,y) = \sin(X) + \cos(Y)$')
plt.colorbar()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Create a surface in matplotlib

# In [ ]
from matplotlib import cm

axes = plt.figure().gca(projection='3d')
axes.plot_surface(xx,xx,Z,cmap=cm.coolwarm)
axes.set_xticks(ticklocs)
axes.set_xticklabels(ticklabels)
axes.set_yticks(ticklocs)
axes.set_yticklabels(ticklabels)

plt.title('$f(x,y) = \\sin(X) + \\cos(Y)$')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 5: Draw the surface in plotly

# In [ ]
import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(x=xx,y=xx,z=Z)])
fig.update_layout(title='$f(x,y) = \sin(X) + \cos(Y)$',
                  autosize=False,height=500)

fig.show()

# In [ ]

# %% [markdown]
# # Exercise 6: Some neat functions to explore

# In [ ]
# sine parameters
sinefreq = .05
phi = np.pi/4
sigma = 3*np.pi
n = 30

# sine wave initializations
xx = np.arange(-n,n+1)
X,Y = np.meshgrid(xx,xx)
U   = X*np.cos(phi) + Y*np.sin(phi)

# create the sine wave and Gaussian
sine2d = np.sin( 2*np.pi*sinefreq*U )
gaus2d = np.exp(-(X**2 + Y**2) / (2*sigma**2))

# point-wise multiply the sine and Gaussian
Z = sine2d * gaus2d

# and plot
fig = go.Figure(data=[go.Surface(x=xx,y=xx,z=Z)])
fig.update_layout(autosize=False)
fig.show()

# In [ ]
# the function
xx = np.linspace(-np.pi,np.pi,40)
X,Y = np.meshgrid(xx,xx)
Z = np.sin( X+Y**2 )

# and plot
fig = go.Figure(data=[go.Surface(x=xx,y=xx,z=Z)])
fig.update_layout(autosize=False)
fig.show()

# In [ ]
# the function
xx = np.linspace(-1,1,40)
X,Y = np.meshgrid(xx,xx)
Z =  X**2 + Y**2 # also try y^3, sqrt

# and plot
fig = go.Figure(data=[go.Surface(x=xx,y=xx,z=Z)])
fig.update_layout(autosize=False,
                  scene = dict(zaxis = dict(range=[0,1])) )
fig.show()

# In [ ]

