# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geometry_surfaceArea.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in geometry
# ### LECTURE: Measuring surfaces of solids
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
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

# In [ ]
# symbolic variable and functions
x  = sym.symbols('x')
fx = x**2 + sym.cos(x**2)
df = sym.diff(fx)

# limits of integration
a = 0
b = 2

# calculate the surface of the solid
integrand = fx * sym.sqrt( 1+df**2 )

# print the integral
display(Math('f(x) \;=\; %s' % (sym.latex(fx))))
display(Math('A \;=\; 2\pi %s' % (sym.latex(sym.Integral(integrand,(x,a,b))))))

# calculate it
surfarea = 2*sym.pi * sym.integrate(integrand,(x,a,b))

# and print it out
print(f'The surface area of the solid is: {surfarea.evalf()}')

# In [ ]

# In [ ]
# re-define the functions (easier than lambdifying for matrix input)
def f(x): return x**2 + np.cos(x**2)

# theta values for revolution
theta = np.linspace(0,2*np.pi,100)

# meshgrid for x and theta
xx = np.linspace(a,b,103)
X, Theta = np.meshgrid(xx,theta)

# Y and Z coordinates for f(x)
Y_f = f(X) * np.cos(Theta)
Z_f = f(X) * np.sin(Theta)


# setup the figure for 3d
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# create the surfaces
ax.plot_surface(X, Y_f, Z_f, color='blue', alpha=.8, edgecolor='none')

# labels etc
ax.set(xlabel='X',ylabel='Y',zlabel='Z')
ax.set_title(f'Solid of revolution (surface area = {surfarea:.2f} a.u.$^2$)')
plt.show()

# In [ ]

