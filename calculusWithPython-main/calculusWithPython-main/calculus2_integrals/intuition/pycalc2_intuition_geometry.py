# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_intuition_geometry.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration and applications
# ## SECTION: Intuition for integration
# ### LECTURE: Integration as geometric area
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202506

# In [ ]
import numpy as np
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
# # Create a function and evaluate it

# In [ ]
# function for the function
def fx(u):
  return u**2 - .5

# In [ ]
# define the grid spacing
dx = .2

# x-axis grid in spacing of dx
xx = np.arange(-1,1+dx,dx)

# evaluate the function at those points
y = fx(xx)

# make a table of x,y pairs
print('   x    |    y')
print('--------|--------')
for xi,yi in zip(xx,y):
  print(f'{xi:>6.3f}  |  {yi:>6.3f}')

# In [ ]

# %% [markdown]
# # Visualize the function and its approximate area

# In [ ]
# plot the function
plt.figure(figsize=(8,5))
plt.plot(xx,y,'ks-',linewidth=2,markersize=10,markerfacecolor=[.7,.3,.9])

# initialize area
area = 0

# plot rectangles
for xi in xx:

  # draw the rectangle
  plt.fill_between([xi-dx/2,xi+dx/2],[fx(xi),fx(xi)],edgecolor='k',facecolor=[.9,.8,.9])

  # sum the area
  area += fx(xi)*dx


# finish the figure
plt.title(f'Area of boxes = {area:.2f}')
plt.xlabel('x')
plt.ylabel('$f(x) = x^2$')
plt.show()

# In [ ]

