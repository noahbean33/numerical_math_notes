# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geometry_CCvolumes.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in geometry
# ### LECTURE: CodeChallenge: Approximating volumes and surfaces
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import Math
import scipy.integrate as spi
from scipy.signal import find_peaks

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Exercise 1: The function and its solid

# In [ ]
x = sym.symbols('x')
fx = sym.log(x)
gx = sym.integrate(fx,x)

display(Math('f(x) \;=\; %s' %sym.latex(fx)))
display(Math('g(x) \;=\; %s' %sym.latex(gx)))

# In [ ]
# find their intersection points

# lambda functions
fx_l = sym.lambdify(x,fx)
gx_l = sym.lambdify(x,gx)


# function difference
xx = np.linspace(.01,4.5,523)
fun_diff = np.abs(fx_l(xx)-gx_l(xx))

# find points closest to zero
peekz = find_peaks(-fun_diff)[0]
intersection_points = xx[peekz]

# and draw
plt.figure(figsize=(8,6))
plt.plot(xx,fx_l(xx),linewidth=2,label=r'$f(x) = %s$' %sym.latex(fx))
plt.plot(xx,gx_l(xx),linewidth=2,label=r'$g(x) = %s$' %sym.latex(gx))

x4area = np.linspace(intersection_points[0],intersection_points[1],79)
plt.fill_between(x4area,fx_l(x4area),gx_l(x4area),color='k',alpha=.5)

plt.plot(intersection_points,fx_l(intersection_points),'ko',markersize=10,markerfacecolor='w')

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(xx[[0,-1]])
plt.ylim(-1.5,2)
plt.legend()
plt.show()

# In [ ]
# minimum function value
min2add = sym.sympify(np.round( np.min(gx_l(xx)) ))

# redefine the functions with a shift
gx = sym.integrate(fx,x) - min2add
fx = sym.log(x) - min2add

fx_l = sym.lambdify(x,fx)
gx_l = sym.lambdify(x,gx)

display(Math('f(x) \;=\; %s' %sym.latex(fx)))
display(Math('g(x) \;=\; %s' %sym.latex(gx)))

# In [ ]
# and draw again
plt.figure(figsize=(8,6))
plt.plot(xx,fx_l(xx),linewidth=2,label=r'$f(x) = %s$' %sym.latex(fx))
plt.plot(xx,gx_l(xx),linewidth=2,label=r'$g(x) = %s$' %sym.latex(gx))
plt.fill_between(x4area,fx_l(x4area),gx_l(x4area),color='k',alpha=.5)
plt.plot(intersection_points,fx_l(intersection_points),'ko',markersize=10,markerfacecolor='w')

plt.gca().set(xlabel='x',ylabel='y',xlim=xx[[0,-1]],ylim=[-.1,3])
plt.legend()
plt.show()

# In [ ]
# bounds
a,b = intersection_points
xx = np.linspace(a,b,200)

# meshgrid for x and theta
X,Theta = np.meshgrid(xx,np.linspace(0,2*np.pi,100))

# get Y and Z coordinates for f(x) and g(x)
Y_f = fx_l(X) * np.cos(Theta)
Z_f = fx_l(X) * np.sin(Theta)

Y_g = gx_l(X) * np.cos(Theta)
Z_g = gx_l(X) * np.sin(Theta)


# setup the figure with a 3D axis
fig = plt.figure(figsize=(12,5))
gs  = fig.add_gridspec(1,2,width_ratios=[2,3])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1],projection='3d')

# plot the 2D shape to revolve
ax1.plot(xx,fx_l(xx),label=r'$f(x)=%s$'%sym.latex(fx))
ax1.plot(xx,np.ones(len(xx))*gx_l(xx),'--',label=r'$g(x)=%s$'%sym.latex(gx))
ax1.fill_between(xx,fx_l(xx),gx_l(xx),color='m',alpha=.2,label='Area to revolve')
ax1.set(xlabel='x',ylabel=r'$y = f(x)$ or $g(x)$',xlim=xx[[0,-1]],title='Function to revolve')
ax1.legend()

# create the surfaces
ax2.plot_surface(X, Y_f, Z_f, alpha=.3)
ax2.plot_surface(X, Y_g, Z_g, alpha=.6)
ax2.set(xlabel='X',ylabel='Y',zlabel='Z',title='Solid in 3D')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Approximate volume and surface area

# In [ ]
# lambda functions for derivatives of the functions (for surface areas)

dfx_l = sym.lambdify(x,sym.diff(fx))
dgx_l = sym.lambdify(x,sym.diff(gx))

display(Math("\\left[ %s \\right]' = %s" %(sym.latex(fx),sym.latex(sym.diff(fx)))))
print('')
display(Math("\\left[ %s \\right]' = %s" %(sym.latex(gx),sym.latex(sym.diff(gx)))))

# In [ ]
# volume via spi.simpson
volume = np.pi * spi.simpson(fx_l(xx)**2 - gx_l(xx)**2,dx=xx[1]-xx[0])

# surface area
surfacearea_f = 2*np.pi * spi.simpson(fx_l(xx) * np.sqrt( 1+dfx_l(xx)**2 ),dx=xx[1]-xx[0])
surfacearea_g = 2*np.pi * spi.simpson(gx_l(xx) * np.sqrt( 1+dgx_l(xx)**2 ),dx=xx[1]-xx[0])

# print
print(f'Volume: {volume:.2f} (au)^3\n')
print(f'Inner surface area: {surfacearea_g:.2f} (au)^2')
print(f'Outer surface area: {surfacearea_f:.2f} (au)^2')
print(f'Total surface area: {surfacearea_g+surfacearea_f:.2f} (au)^2')

# In [ ]

# %% [markdown]
# # Exercise 3: A solid face

# In [ ]
# xy coordinates
y = np.array([1,0.981328,0.975104,0.973029,0.96888,0.960581,0.952282,0.943983,0.93361,0.925311,0.917012,0.906639,0.892116,0.875519,0.860996,0.844398,0.817427,0.804979,0.788382,0.746888,0.73029,0.717842,0.688797,0.6639,0.647303,0.624481,0.605809,0.589212,0.562241,0.545643,0.529046,0.506224,0.495851,0.481328,0.477178,0.462656,0.452282,0.43361,0.423237,0.410788,0.404564,0.39834,0.394191,0.390041,0.387967,0.383817,0.377593,0.363071,0.350622,0.342324,0.325726,0.311203,0.302905,0.275934,0.261411,0.246888,0.23029,0.211618,0.19917,0.190871,0.174274,0.16805,0.157676,0.149378,0.141079,0.13278,0.126556,0.116183,0.103734,0.0871369,0.0726141,0.060166,0.0435685,0.026971,0.0124481,0])
xR = np.array([1, 0.96094299, 0.91765732, 0.87437165, 0.80078537, 0.7574997 , 0.7272, 0.6969003 , 0.65794329, 0.62764268, 0.58868567, 0.55405732, 0.50644299, 0.46748597, 0.42852896, 0.38957104, 0.33762896, 0.31598567, 0.3029997 , 0.28135732, 0.2727    , 0.2727    , 0.28135732, 0.28568597, 0.27702866, 0.25971403, 0.22941433, 0.20344329, 0.13851433, 0.1168714 , 0.09522857, 0.0909    , 0.1168714 , 0.13851433, 0.14717146, 0.16448573, 0.1688143 , 0.14717146, 0.1298571 , 0.12552854, 0.13851433, 0.15150003, 0.16448573, 0.17314287, 0.17314287, 0.16448573, 0.15150003, 0.14717146, 0.16015716, 0.17747143, 0.20777104, 0.22508567, 0.22508567, 0.2120997 , 0.19911463, 0.1818    , 0.17314287, 0.1818    , 0.2120997 , 0.23374299, 0.32464299, 0.36792866, 0.44584268, 0.51077165, 0.56704329, 0.61898537, 0.6665997 , 0.70122896, 0.73585732, 0.77914299, 0.82242866, 0.8483997 , 0.87004299, 0.89601403, 0.91332866,0.93064329])
xL = np.array([-1, -0.96094299, -0.91765732, -0.87437165, -0.80078537,-0.7574997 , -0.7272 , -0.6969003 , -0.65794329, -0.62764268,-0.58868567, -0.55405732, -0.50644299, -0.46748597, -0.42852896,-0.38957104, -0.33762896, -0.31598567, -0.3029997 , -0.28135732,-0.2727    , -0.2727    , -0.28135732, -0.28568597, -0.27702866,-0.25971403, -0.22941433, -0.20344329, -0.13851433, -0.1168714 ,-0.09522857, -0.0909    , -0.1168714 , -0.13851433, -0.14717146,-0.16448573, -0.1688143 , -0.14717146, -0.1298571 , -0.12552854,-0.13851433, -0.15150003, -0.16448573, -0.17314287, -0.17314287,-0.16448573, -0.15150003, -0.14717146, -0.16015716, -0.17747143,-0.20777104, -0.22508567, -0.22508567, -0.2120997 , -0.19911463,-0.1818    , -0.17314287, -0.1818    , -0.2120997 , -0.23374299,-0.32464299, -0.36792866, -0.44584268, -0.51077165, -0.56704329,-0.61898537, -0.6665997 , -0.70122896, -0.73585732, -0.77914299,-0.82242866, -0.8483997 , -0.87004299, -0.89601403, -0.91332866,-0.93064329])

# In [ ]
# forward x-axis for positive volume/area
y = y[::-1]
xL= xL[::-1]

# In [ ]
# volume via spi.simpson
volume = np.pi * spi.simpson((xL+1)**2,x=y)

# surface area
surfacearea = 2*np.pi * spi.simpson((xL+1) * np.sqrt( 1+np.gradient(xL+1)**2 ),x=y)

volume,surfacearea

# In [ ]
# bounds
a,b = 0,2

# meshgrid for x and theta
X,Theta = np.meshgrid(y,np.linspace(0,2*np.pi,100))

# get Y and Z coordinates for f(x) and g(x)
Y_f = (xL+1) * np.cos(Theta)
Z_f = (xL+1) * np.sin(Theta)


# setup the figure with a 3D axis
fig = plt.figure(figsize=(12,4))
gs  = fig.add_gridspec(1,2,width_ratios=[2,3])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1],projection='3d')

# plot the 2D shape to revolve
ax1.plot(y,(xL+1))
ax1.set(xlabel='x',ylabel=r'$y = f(x)$',title='Function to revolve')

# create the surfaces
ax2.plot_surface(X, Y_f, Z_f, alpha=.3)
ax2.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'Volume = {volume:.3f}\nSurface area = {surfacearea:.3f}')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Faces in plotly

# In [ ]
# using plotly
import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(x=X, y=Y_f, z=Z_f, colorscale='Blues', opacity=.8)])
fig.show()

# In [ ]

