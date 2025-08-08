# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geometry_CCnumericalApprox.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in geometry
# ### LECTURE: CodeChallenge: Numerical approximations
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import display,Math
import scipy.integrate as spi

# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Exercise 1: Initial exploration in sympy

# In [ ]
# variable, function, integral
x = sym.symbols('x')

fx = sym.log(x)
gx = sym.integrate(fx)

sym.plot(fx,gx,(x,.001,4.5),ylim=[-1.5,2])

display(Math('f(x) = %s' %sym.latex(fx)))
display(Math('g(x) = %s' %sym.latex(gx)))

# In [ ]
# find the intersection points
sym.solve(fx-gx)

# In [ ]

# %% [markdown]
# # Exercise 2: Transform to numpy and visualize

# In [ ]
# lambdify the functions
fx_l = sym.lambdify(x,fx)
gx_l = sym.lambdify(x,gx)
diff_l = sym.lambdify(x,fx-gx) # used in exercise 4

# evaluate the functions
xx = np.linspace(.01,4.5,123)


# and draw
plt.figure(figsize=(8,6))
plt.plot(xx,fx_l(xx),linewidth=2,label=r'$f(x) = %s$' %sym.latex(fx))
plt.plot(xx,gx_l(xx),linewidth=2,label=r'$g(x) = %s$' %sym.latex(gx))

plt.gca().set(xlabel='x',ylabel='y',xlim=xx[[0,-1]],ylim=[-1.5,2])
plt.legend()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Approximate function intersections

# In [ ]
# function difference
fun_diff = np.abs(fx_l(xx)-gx_l(xx))

plt.plot(xx,fun_diff)
plt.ylim(0,1.5)
plt.xlabel('x')
plt.ylabel('|f-g|')
plt.show()

# In [ ]
# find points closest to zero
from scipy.signal import find_peaks

peekz = find_peaks(-fun_diff)[0]
intersection_points = xx[peekz]

print(intersection_points)

# In [ ]
# and draw
plt.figure(figsize=(8,6))
plt.plot(xx,fx_l(xx),linewidth=2,label=r'$f(x) = %s$' %sym.latex(fx))
plt.plot(xx,gx_l(xx),linewidth=2,label=r'$g(x) = %s$' %sym.latex(gx))

x4area = np.linspace(intersection_points[0],intersection_points[1],79)
plt.fill_between(x4area,fx_l(x4area),gx_l(x4area),color='k',alpha=.5)

plt.plot(intersection_points,fx_l(intersection_points),'ko',markersize=10,markerfacecolor='w')

plt.gca().set(xlabel='x',ylabel='y',xlim=xx[[0,-1]],ylim=[-1.5,2])
plt.legend()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Numerical approximation using scipy

# In [ ]
# empirical areas
fx_int,_ = spi.quad(fx_l, intersection_points[0],intersection_points[1])
gx_int,_ = spi.quad(gx_l, intersection_points[0],intersection_points[1])
diff_int,_ = spi.quad(diff_l, intersection_points[0],intersection_points[1])

display(Math('%s \\approx %s' %(sym.latex(sym.Integral(fx,(x,np.round(intersection_points[0],2),np.round(intersection_points[1],2)))),np.round(fx_int,2))))
print('')
display(Math('%s \\approx %s' %(sym.latex(sym.Integral(gx,(x,np.round(intersection_points[0],2),np.round(intersection_points[1],2)))),np.round(gx_int,2))))
print('')
display(Math('%s \\approx %s' %(sym.latex(sym.Integral(fx-gx,(x,np.round(intersection_points[0],2),np.round(intersection_points[1],2)))),np.round(diff_int,2))))

# In [ ]

