# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_integrate_CCareasAlgorithm.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integrating functions
# ### LECTURE: CodeChallenge: Net and total areas algorithm
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

# %% [markdown]
# # Exercise 1: Graph the function and bounds

# In [ ]
x = sym.symbols('x')

# create a function
fun = -x**3 + 3*x**2 - 2*x

# define integration bounds
a,b = -1,2


# -- for Exercise 3 -- #
# fun = sym.sqrt(x) - x**2
# a,b = 0,2

# quick-n-dirty plot
sym.plot(fun,(x,-2,4),ylim=[-3,3]);

# In [ ]
# find the roots
roots = sym.solve(fun)

# segmentation breakpoints are the roots and the bounds
segbounds = np.unique(np.append(roots,(a,b)))

# define x-axis grid and lambdify the function
xx = np.linspace(a-.25,b+.25,517)
fun_l = sym.lambdify(x,fun)


plt.figure(figsize=(12,5))

# plot the function graph, the y=0 line, and lines for the integration bounds
plt.plot(xx,fun_l(xx),'k',linewidth=2,label=r'$f(x) = %s$' %sym.latex(fun))
plt.axhline(0,color='lightgray',linestyle='--',zorder=-3)
plt.axvline(a,color='lightblue',linestyle='--',zorder=-3)
plt.axvline(b,color='lightblue',linestyle='--',zorder=-3)

# plot the roots and areas
for segi in range(len(segbounds)):

  # draw and label the dot
  bound = float( segbounds[segi] )
  plt.plot(bound,fun_l(bound),'o',markersize=10,label=r'$f(%g)=%g$' %(bound,fun_l(bound)))

  # draw a patch around the area
  if segi<len(segbounds)-1:

    # find x values for this segment
    xxSegment = xx[(xx>bound) & (xx<segbounds[segi+1])]

    # pick a color based on sign
    c = 'g' if np.mean(fun_l(xxSegment))>0 else 'r'

    # draw the patch
    plt.fill_between(xxSegment,fun_l(xxSegment),color=c,alpha=.2)


plt.gca().set(xlim=xx[[0,-1]],ylim=[-1,7],xlabel='x',ylabel='y = f(x)')
plt.legend()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Compute segment, net, and total areas

# In [ ]
# initialize areas for each segment
segmentArea = np.zeros(len(segbounds)-1)

# loop over segments
for segi in range(len(segbounds)-1):

  boundL = segbounds[segi]
  boundR = segbounds[segi+1]

  # compute the area in this segment
  segmentArea[segi] = sym.integrate(fun,(x,boundL,boundR))

  # print the results
  display(Math('%s = %s' %(sym.latex(sym.Integral(fun,(x,boundL,boundR))),segmentArea[segi])))
  print('')


# now print net and total area
print(f'Net area   = {np.sum(segmentArea)}')
print(f'Total area = {np.sum(np.abs(segmentArea))}')

# In [ ]

