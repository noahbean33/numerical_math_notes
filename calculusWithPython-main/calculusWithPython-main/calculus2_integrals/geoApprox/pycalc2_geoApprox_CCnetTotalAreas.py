# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geoApprox_CCnetTotalAreas.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Geometric approximations
# ### LECTURE: CodeChallenge: Net and total area
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

# %% [markdown]
# # Exercise 1: Approximate net/total areas using trapezoids

# In [ ]
# function for the function
def fx(u):
  return u**2 - .5

# In [ ]
# create deltax
n = 6
a = -.5
b = 1
deltax = (b-a)/n

breakPoints = [ a+deltax*i for i in range(n+1) ]


# plot the function
_,axs = plt.subplots(1,figsize=(7,4))

# plot the function
xx = np.linspace(a-.1,b+.1,301)
axs.plot(xx,fx(xx),'r',markersize=10,markerfacecolor=[.7,.3,.9])

# initialize areas
areaNet = 0
areaTot = 0

# plot rectangles
for i in range(n):

  bp = breakPoints[i]
  bp += deltax/2 # shift breakpoint by deltax/2

  # compute the function value at both edges
  yL = fx(bp-deltax/2)
  yR = fx(bp+deltax/2)

  # draw the rectangle
  faCo=[.9,.5,.5] if yL+yR<0 else [.5,.9,.4]
  axs.fill_between([bp-deltax/2,bp+deltax/2],[yL,yR],edgecolor='k',facecolor=faCo)

  # sum the area
  areaNet += ( (yL+yR)/2 ) * deltax
  areaTot += np.abs( (yL+yR)/2 ) * deltax

# set the labels (*after* the for-loop)
axs.set(xlabel='x',ylabel=r'$y = x^2-.5$')
axs.set_title(r'Net area = %.3f, total area = %.3f $\Delta$x=%g' %(areaNet,areaTot,deltax),wrap=True)


# finalize
plt.axhline(0,color='gray',linestyle='--')
plt.axvline(a,color='gray',linestyle='--')
plt.axvline(b,color='gray',linestyle='--')
plt.xlim(xx[[0,-1]])
plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Exact areas using sympy

# In [ ]
x = sym.symbols('x')
expr = x**2 - .5

# net area
areaNet_s = sym.integrate(expr,(x,a,b))
areaTot_s = sym.integrate(sym.Abs(expr),(x,a,b))

print(f'Sympy: Net area   = {areaNet_s:.4f}')
print(f'Numpy: Net area   = {areaNet:.4f}')
print('')

print(f'Sympy: Total area = {areaTot_s:.4f}')
print(f'Numpy: Total area = {areaTot:.4f}')

# In [ ]

# %% [markdown]
# # Exercise 3: Visualize using the sympy plotting engine

# In [ ]
# create plot objects
p1 = sym.plot(expr,(x,-2,2), show=False,line_color='blue',label=r'$x^2-.5$')
p2 = sym.plot(sym.Abs(expr),(x,-2,2), show=False,line_color='red',label=r'$|x^2-.5|$')

# combine them
p1.extend(p2)

# Set some additional properties for the plot
p1.xlabel = 'x'
p1.ylabel = 'f(x)'
p1.legend = True

# Show the combined plot
p1.show()

# In [ ]

