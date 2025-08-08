# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_multivar_defIntegration.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Multivariable integration
# ### LECTURE: Multivariable definite integrals
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import display,Math

# scipy integration
from scipy import integrate

# used to draw the bounding box for integration
import matplotlib.patches as patches

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Create the function in sympy and numpy

# In [ ]
# the variables
x,y = sym.symbols('x,y')

# and the function
f_xy = 10 - ( (x**2-y**2)/8 )

# need to discretize it
f_xy_l = sym.lambdify((x,y),f_xy)

# In [ ]
# evaluate the function on a discrete grid

# function domains for visualizations
xx = np.linspace(-10,10,41)
yy = np.linspace(-10,10,41)

# for later, we'll need deltax and deltay
dx = xx[1] - xx[0]
dy = yy[1] - yy[0]

# need a grid of points, not two vectors
XX,YY = np.meshgrid(xx,yy)
Z = f_xy_l(XX,YY)

# In [ ]
# show the graph of the function
_,ax = plt.subplots(1,figsize=(8,8))
ax.imshow(Z,extent=[xx[0], xx[-1], yy[-1], yy[0]],aspect='auto')

ax.set(xlabel='X',ylabel='Y')
ax.set_title(r'$f(x,y) = %s$' %sym.latex(f_xy))
plt.show()

# In [ ]

# %% [markdown]
# # Calculate the definite integral using sympy

# In [ ]
# integration bounds
x_a,x_b = 0,5
y_a,y_b = 0,4

# partial and full integral
area_x = sym.integrate(f_xy,(x,x_a,x_b))
area = sym.integrate(f_xy,(x,x_a,x_b),(y,y_a,y_b))

# and report
display(Math('%s = %s' %(sym.latex(sym.Integral(f_xy,(x,x_a,x_b),y)),sym.latex(area_x))))
print('')
display(Math('%s = %s = %s' %(sym.latex(sym.Integral(f_xy,(x,x_a,x_b),(y,y_a,y_b))),sym.latex(area),sym.latex(area.evalf()))))

# In [ ]

# %% [markdown]
# # Definite integral using numpy

# In [ ]
# show the graph again with a box for the integral
_,ax = plt.subplots(1,figsize=(8,8))
ax.imshow(Z,extent=[xx[0], xx[-1], yy[-1], yy[0]],aspect='auto')

# create and draw a rectangle corresponding to the integration bounds
rect = patches.Rectangle((x_a, y_a), x_b-x_a, y_b-y_a, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

ax.set(xlabel='X',ylabel='Y')
ax.set_title(r'$f(x,y) = %s$' %sym.latex(f_xy))
plt.show()

# In [ ]
# first, find the indices matching the bounds
x_a_idx = np.argmin( np.abs(xx-x_a) )
x_b_idx = np.argmin( np.abs(xx-x_b) )
y_a_idx = np.argmin( np.abs(yy-y_a) )
y_b_idx = np.argmin( np.abs(yy-y_b) )

# then, extract the function surface from those bounds
Z_sub = Z[y_a_idx:y_b_idx,x_a_idx:x_b_idx]
print(f'Size of function landscape: {Z.shape}')
print(f'Size of function subset: {Z_sub.shape}')
print('')

# finally, sum and scale
area_x_np = np.sum(Z_sub,axis=1) * dx
area_np = np.sum(Z_sub) * dx * dy

# and print out
display(Math('%s \\approx %s' %(sym.latex(sym.Integral(f_xy,(x,x_a,x_b),y)),sym.latex(area_x_np))))
print('')
display(Math('%s \\approx %s' %(sym.latex(sym.Integral(f_xy,(x,x_a,x_b),(y,y_a,y_b))),sym.latex(area_np))))

# In [ ]

# %% [markdown]
# # Definite integral using scipy

# In [ ]
# scipy.integrate.dblquad needs the variables swapped (y,x)
def f_xy_sp(y,x):
  return 10 - (x**2 - y**2) / 8

# Calculate the double integral (inner variable first)
area_sp,_ = integrate.dblquad(f_xy_sp, x_a,x_b, lambda x: y_a, lambda x: y_b)

display(Math('%s \\approx %s' %(sym.latex(sym.Integral(f_xy,(x,x_a,x_b),(y,y_a,y_b))),sym.latex(area_sp))))

# In [ ]

