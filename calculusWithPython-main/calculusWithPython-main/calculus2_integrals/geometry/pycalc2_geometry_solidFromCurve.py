# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geometry_solidFromCurve.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in geometry
# ### LECTURE: Creating volumes of revolution
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
# import libraries and define global settings
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# In [ ]
# define functions
x = sym.symbols('x')

# example 1
fx = abs(sym.sin(x**2))
gx = 0

# example 2
# fx = sym.sqrt(x)
# gx = 0

# # example 3
# fx = abs(sym.log(x)) + .1
# gx = 0

# example 4
# fx = sym.sqrt(x)
# gx = -sym.cos(x)+1


# define the functions
f = sym.lambdify(x,fx)
g = sym.lambdify(x,gx)

# bounds
a,b = 0,2
xx = np.linspace(a,b,100)

# meshgrid for x and theta
X,Theta = np.meshgrid(xx,np.linspace(0,2*np.pi,100))

# get Y and Z coordinates for f(x) and g(x)
Y_f = f(X) * np.cos(Theta)
Z_f = f(X) * np.sin(Theta)

Y_g = g(X) * np.cos(Theta)
Z_g = g(X) * np.sin(Theta)


# setup the figure with a 3D axis
fig = plt.figure(figsize=(12,5))
gs  = fig.add_gridspec(1,2,width_ratios=[2,3])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1],projection='3d')

# plot the 2D shape to revolve
ax1.plot(xx,f(xx),'k',label=r'$f(x)=%s$'%sym.latex(fx))
ax1.plot(xx,np.ones(len(xx))*g(xx),'r--',label=r'$g(x)=%s$'%sym.latex(gx))
ax1.fill_between(xx,f(xx),g(xx),alpha=.2,label='Area to revolve')
ax1.set(xlabel='x',ylabel=r'$y = f(x)$ or $g(x)$',xlim=xx[[0,-1]])
ax1.legend()

# create the surfaces
ax2.plot_surface(X, Y_f, Z_f, alpha=.3)
ax2.plot_surface(X, Y_g, Z_g, alpha=.6)
ax2.set(xlabel='X',ylabel='Y',zlabel='Z')

plt.tight_layout()
plt.show()

# In [ ]

# In [ ]
# using plotly
import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(x=X, y=Y_f, z=Z_f, colorscale='Blues', opacity=.8)])
fig.add_trace( go.Surface(x=X, y=Y_g, z=Z_g, colorscale='Reds', opacity=.5) )
fig.show()

# In [ ]

