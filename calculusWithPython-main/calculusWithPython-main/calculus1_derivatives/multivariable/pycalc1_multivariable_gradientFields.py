# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_multivariable_gradientFields.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Multivariable differentiation
# ### LECTURE: CodeChallenge: Gradient fields
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
# # Exercise 1: The function and its partials

# In [ ]
# create variables x and y
x,y = sym.symbols('x,y')

# the function
fxy = x * sym.exp( -(x**2+y**2) )
# fxy = sym.sin(x**2+y) + sym.cos(y)
# fxy = sym.exp(-sym.Abs(x*y)) # works in Real domain

# a quick visualization
sym.plotting.plot3d(fxy,(x,-3,3),(y,-3,3));

# In [ ]
# lambdify the function and its partial derivatives
fxy_l = sym.lambdify((x,y),fxy)
dfx_l = sym.lambdify((x,y),sym.diff(fxy,x))
dfy_l = sym.lambdify((x,y),sym.diff(fxy,y))

N = 21
xx = np.linspace(-3,3,N)
X,Y = np.meshgrid(xx,xx)

Z  = fxy_l(X,Y)
Zx = dfx_l(X,Y)
Zy = dfy_l(X,Y)

_,axs = plt.subplots(1,3,figsize=(8,5))
axs[0].imshow(Z,origin='lower',extent=[xx[0],xx[-1],xx[0],xx[-1]])
axs[0].set_title(f'$f(x,y) = {sym.latex(fxy)}$')

axs[1].imshow(Zx,origin='lower',extent=[xx[0],xx[-1],xx[0],xx[-1]])
axs[1].set_title(f'$f_x = {sym.latex(sym.diff(fxy,x))}$')

axs[2].imshow(Zy,origin='lower',extent=[xx[0],xx[-1],xx[0],xx[-1]])
axs[2].set_title(f'$f_y = {sym.latex(sym.diff(fxy,y))}$')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Gradient field

# In [ ]
# plot the gradient
plt.figure(figsize=(5,5))
plt.contourf(xx,xx,Z,40)
plt.quiver(xx,xx,Zx,Zy);

# In [ ]

# %% [markdown]
# # Exercise 3: Repeat using np.gradient

# In [ ]
# numpy's empirical calulation of the gradient
gradx,grady = np.gradient(Z)

plt.figure(figsize=(5,5))
plt.contourf(xx,xx,Z,40)
plt.quiver(xx,xx,grady,gradx);

# In [ ]

# %% [markdown]
# # Exercise 5: Climb to the peak

# In [ ]
# start at a random loc
# loc = np.array([12,15]) # note: fixing the start loc is useful for development
loc = np.random.choice(np.arange(N),2)

nIterations = 10

# combine gradient maps for convenience
G = np.concatenate((gradx[None],grady[None]),axis=0)

plt.imshow(Z,origin='lower',extent=[xx[0],xx[-1],xx[0],xx[-1]])
# plt.quiver(xx,xx,grady,gradx)
plt.plot(xx[loc[1]],xx[loc[0]],'ks')

# algorithm loop
for i in range(nIterations):
  
  # find maximum directions
  dirXY = np.argmax(np.abs(G[:,loc[0],loc[1]]))
  dirUD = np.sign(G[dirXY,loc[0],loc[1]])

  # move the particle along that direction
  if dirXY==0:
    loc[0] += dirUD
  elif dirXY==1:
    loc[1] += dirUD

  # possibly fix boundaries
  loc[0] = np.max((0,loc[0]))
  loc[0] = np.min((N-1,loc[0]))
  
  loc[1] = np.max((0,loc[1]))
  loc[1] = np.min((N-1,loc[1]))

  # update the plot
  plt.plot(xx[loc[1]],xx[loc[0]],'o',color=[np.sqrt(i/nIterations),.2,.2])


# pink diamond at the end of the trajectory
plt.plot(xx[loc[1]],xx[loc[0]],'d',color='pink')
plt.show()

# In [ ]

