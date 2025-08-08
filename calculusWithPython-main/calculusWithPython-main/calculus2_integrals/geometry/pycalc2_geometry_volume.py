# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geometry_volume.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in geometry
# ### LECTURE: Measuring volumes of solids
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
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
# symbolic variable and functions
x  = sym.symbols('x')

# disk method
fx = sym.sqrt(x)
gx = sym.sympify(0)

# washer method
# fx = x**3 + 1
# gx = x + 1

# limits of integration
a,b = 0,1

# calculate the volume of revolution
volume = sym.pi * sym.integrate(fx**2 - gx**2, (x,a,b))

# and print it out (negative volume??)
print(f'The volume of the solid of revolution is: {volume.evalf()} (a.u.)^3')

# In [ ]
# Generate x values
xx = np.linspace(a,b,100)

# Generate y values for both functions (and convert from sympy to float)
f_y = np.array([float(fx.subs(x,xi)) for xi in xx])
g_y = np.array([float(gx.subs(x,xi).evalf()) for xi in xx])

# Plot the functions
_,ax = plt.subplots(figsize=(8,6))
ax.fill_between(xx, f_y, g_y, color='lightblue', label='Area between f(x) and y=0')
ax.plot(xx, f_y, label=r'$f(x) = %s$' %sym.latex(fx))
# ax.plot(xx, g_y, label=r'$g(x) = %s$' %sym.latex(gx))
ax.legend()

# Labels and title
ax.set(xlabel='x',ylabel='y',title='Area to revolve around the x-axis')
plt.show()

# In [ ]
# re-define the functions (alternative to lambdifying for matrix input)

# disk method
def f(x): return np.sqrt(x)
def g(x): return 0

# washer method
# def f(x): return x**3 + 1
# def g(x): return x + 1

# theta values for revolution
theta = np.linspace(0,2*np.pi,100)

# meshgrid for x and theta
X, Theta = np.meshgrid(xx,theta)

# Y and Z coordinates for f(x) and g(x)
Y_f = f(X) * np.cos(Theta)
Z_f = f(X) * np.sin(Theta)

Y_g = g(X) * np.cos(Theta)
Z_g = g(X) * np.sin(Theta)


# setup the figure for 3d
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# create the surfaces
ax.plot_surface(X, Y_f, Z_f, color='red', alpha=.8, edgecolor='none')
ax.plot_surface(X, Y_g, Z_g, color='lightblue',  alpha=.5, edgecolor='none')

# labels etc
ax.set(xlabel='X',ylabel='Y',zlabel='Z')
ax.set_title(f'Volume of solid = {volume:.3f} (a.u.)$^3$')
plt.show()

# In [ ]

