# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geometry_calculartLissajous.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in geometry
# ### LECTURE: Calculart: Lissajous's hypotrochoids
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import scipy.integrate as spi
import matplotlib.pyplot as plt

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Calculart 1: Lissajous curves

# In [ ]
# parameters
A = 2
alpha = np.pi/2
gamma = .1

B = 2
beta = np.pi

# function
t = np.linspace(0,2*np.pi,598)
x = A*np.sin(alpha*t + gamma)
y = B*np.sin(beta*t)

# plot
plt.plot(x,y)
plt.gca().set(xlabel='$x(t)$',ylabel='$y(t)$',
              title=f'$\\bf{{Lissajous\; curve}}$\nA = {A}, B = {B},\n $\\alpha$ = {alpha:.2f}, $\\beta$ = {beta:.2f}, $\\gamma$ = {gamma:.2f}')
plt.show()

# In [ ]

# %% [markdown]
# # Calculart 2: Lissajous' lengths

# In [ ]
def lengthFun(x,y,t):
  dx_dt = np.gradient(x,t)
  dy_dt = np.gradient(y,t)
  return spi.simpson(np.sqrt(dx_dt**2 + dy_dt**2),x=t)

# In [ ]
# parameters
A = 2
alpha = np.pi/2
gammas = np.linspace(0,.5,15)

B = 2
beta = np.pi

# initializations
t = np.linspace(0,2*np.pi,598)
L = np.zeros(len(gammas))

colors = np.linspace([.4,0,.2],[1,.4,.5],len(gammas))

# loop!
_,axs = plt.subplots(1,2,figsize=(12,4))

for idx,gamma in enumerate(gammas):
  x = A*np.sin(alpha*t + gamma)
  y = B*np.sin(beta*t)
  axs[0].plot(x,y,color=colors[idx])

  # length of this curve
  L[idx] = lengthFun(x,y,t)

# plot labels
axs[0].set(xlabel='$x(t)$',ylabel='$y(t)$',title='Lissajous curves')

# plot the lengths
axs[1].scatter(gammas,L,c=colors,s=100)
axs[1].set(xlabel=r'$\gamma$',ylabel='Curve length',title=r'Lengths by $\gamma$')

plt.tight_layout()
plt.show()

# In [ ]
A = 2
alphas = np.linspace(2,2.1,15)
gamma = 0

B = 2
beta = np.pi

t = np.linspace(0,2*np.pi,598)
L = np.zeros(len(alphas))

_,axs = plt.subplots(1,2,figsize=(12,4))

for idx,alpha in enumerate(alphas):
  x = A*np.sin(alpha*t + gamma)
  y = B*np.sin(beta*t)
  axs[0].plot(x,y,color=colors[idx])

  # length of this curve
  L[idx] = lengthFun(x,y,t)


axs[0].set(xlabel='$x(t)$',ylabel='$y(t)$',title='Lissajous curves')

axs[1].scatter(alphas,L,c=colors,s=100)
axs[1].set(xlabel='$\\alpha$',ylabel='Curve length',title=r'Lengths by $\alpha$')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Calculart 3: Multiparameter space

# In [ ]
A = 2
alphas = np.linspace(2,2.1,15)
gammas = np.linspace(0,.5,25)

B = 2
beta = np.pi

t = np.linspace(0,2*np.pi,598)
L = np.zeros((len(alphas),len(gammas)))

_,axs = plt.subplots(1,2,figsize=(12,4))

for i,gamma in enumerate(gammas):
  for j,alpha in enumerate(alphas):

    # create and measure this curve
    x = A*np.sin(alpha*t + gamma)
    y = B*np.sin(beta*t)
    L[j,i] = lengthFun(x,y,t)

    # plotting
    RGB = [i/len(gammas), .4, 1-j/len(alphas)]
    axs[0].plot(x,y,color=RGB,linewidth=.5)


h = axs[1].imshow(L,extent=[gammas[0],gammas[-1],alphas[0],alphas[-1]],origin='lower',aspect='auto')
axs[1].set(xlabel=r'$\gamma$',ylabel=r'$\alpha$',title='Matrix of lengths by parameters')
plt.colorbar(h)
axs[0].set(xlabel='$x(t)$',ylabel='$y(t)$',title='Lots of Lissajous curves')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Calculart 4: One hypotrochoid

# In [ ]
# parameters
R = 4
r = 3.5
d = 1.5

# function
theta = np.linspace(0,14*np.pi,435)
x = (R-r)*np.cos(theta) + d*np.cos((R-r)*theta/r)
y = (R-r)*np.sin(theta) - d*np.sin((R-r)*theta/r)

# length
L = lengthFun(x,y,theta)

# plot
plt.plot(x,y,'k',linewidth=2)
plt.gca().set(xlabel='$x(t)$',ylabel='$y(t)$',title=f'$\\bf{{Hypotrochoid}}$ (length = {L:.3f})\nR = {R}, r = {r}, d = {d}')
plt.show()

# In [ ]
# alternative option for plotting

# create circular color pallette
th = np.linspace(0,2*np.pi,len(theta))

Red   = (np.cos(th)+1)/2
Green = np.full(len(theta),.4)
Blue  = (np.sin(th)+1)/2

colors = np.vstack( (Red,Green,Blue) ).T

# and plot
plt.scatter(x,y,s=50,c=colors,alpha=.5)
plt.gca().set(xlabel='$x(t)$',ylabel='$y(t)$',title=f'$\\bf{{Hypotrochoid}}$ (length = {L:.3f})\nR = {R}, r = {r}, d = {d}')
plt.show()

# In [ ]

# %% [markdown]
# # Calculart 5: A hypotrochoid family portrait

# In [ ]
# parameters
R = 4
rs = np.linspace(R-2,R+3,20)
d = 1.5

theta = np.linspace(0,4*np.pi,435)
lengths = np.zeros(len(rs))

fig,axs = plt.subplots(1,2,figsize=(10,5))

linecolors = np.linspace([.4,.3,1],[1,.4,.2],len(rs))

for idx,r in enumerate(rs):

  # function
  x = (R-r)*np.cos(theta) + d*np.cos((R-r)/r*theta)
  y = (R-r)*np.sin(theta) - d*np.sin((R-r)/r*theta)

  # length
  lengths[idx] = lengthFun(x,y,theta)

  # plot
  axs[0].plot(x,y,linewidth=1.5,color=linecolors[idx],alpha=.5)


axs[1].scatter(rs-R,lengths,c=linecolors,s=80)
axs[1].set(xlabel='$r-R$',ylabel='Length',title='Lengths by $r-R$')
axs[0].set(xlabel='$x(t)$',ylabel='$y(t)$',title=f'Many curves (R = {R})')

plt.tight_layout()
plt.show()

# In [ ]

