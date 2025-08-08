# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geoApprox_CCriemann.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Geometric approximations
# ### LECTURE: CodeChallenge: Riemann approximations
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
# # Exercise 1: Create function, bounds, and $\Delta$x

# In [ ]
# function for the function
def fx(x):
  return np.exp(x)/10 + np.cos(x)

# In [ ]
# specify bounds
a = -.5
b = np.pi

# create deltax
n = 14
deltax = (b-a)/n

breakPointsL = np.array([ a+deltax*i for i in range(n+1) ])
breakPoints = np.linspace(a,b,n+1)

# confirmation...
print('Breakpoints from formula:')
print(breakPointsL)

print('\nBreakpoints from np.linspace:')
print(breakPoints)

print('\n\n\Delta-x from formula:')
print(deltax)
print('\n\Delta-x from np.linspace:')
print(np.diff(breakPoints))

# In [ ]
# x-axis spacing
xx = np.linspace(-1,3.4,909)

# show the function, bounds, area, and breakpoints
plt.figure(figsize=(10,5))
plt.plot(xx,fx(xx),'k',linewidth=2,label=r'$f(x) = \cos(x) + e^{x}/10$')

plt.axvline(a,color=[.4,.9,.2],linestyle='--',label=f'a = {a:.2f}')
plt.axvline(b,color=[.9,.5,.5],linestyle='--',label=f'b = {b:.2f}')
plt.axhline(0,color=[.8,.8,.8],linestyle='--',label='y = 0')

plt.fill_between(xx[(xx>a) & (xx<b)],fx(xx[(xx>a) & (xx<b)]),color='m',alpha=.4,edgecolor=None,label='Area')

for bp in breakPoints:
  plt.plot([bp,bp],[-.05,.05],'k')

plt.legend()
plt.gca().set(xlabel='x',ylabel='$y=f(x)$',xlim=xx[[0,-1]],ylim=[-.05,1.5])
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Compute Riemann sums and compare with the integral

# In [ ]
# Riemann sums

# left rule
area_left = np.sum(fx( breakPoints[:-1] )) * deltax

# right rule
area_right = np.sum(fx( breakPoints[1:] )) * deltax

# midpoint rule
area_midpoint = np.sum(fx( (breakPoints[:-1]+breakPoints[1:])/2 )) * deltax

# In [ ]
# true integral using sympy
t = sym.symbols('t')
area_analytic = sym.integrate( sym.exp(t)/10 + sym.cos(t),(t,a,b))

# print all results
print(f'    Using left rule: {area_left:.8f}')
print(f'   Using right rule: {area_right:.8f}')
print(f'Using midpoint rule: {area_midpoint:.8f}')
print(f'  Definite integral: {area_analytic:.8f}')

# In [ ]

# %% [markdown]
# # Exercise 3: Visualize the partitions

# In [ ]
# show the function, bounds, area, and breakpoints
_,axs = plt.subplots(1,3,figsize=(18,4))


# same for all plots
for ax in axs:
  ax.plot(xx,fx(xx),'k',linewidth=2,label=r'$f(x) = \cos(x) + e^{x}/10$')
  ax.axvline(a,color=[.4,.9,.2],linestyle='--',label=f'a = {a:.2f}')
  ax.axvline(b,color=[.9,.5,.5],linestyle='--',label=f'b = {b:.2f}')
  ax.legend(fontsize=11)
  ax.set(xlabel='x',ylabel='y=f(x)',xlim=xx[[0,-1]],ylim=[0,1.5])


# now for the bars
for i in range(n):

  # bars for left rule
  bp = breakPoints[i]
  axs[0].fill_between([bp,bp+deltax],[fx(bp),fx(bp)],color='m',alpha=.4)

  # bars for right rule
  axs[1].fill_between([bp,bp+deltax],[fx(bp+deltax),fx(bp+deltax)],color='m',alpha=.4)

  # bars for midpoint rule
  bp += deltax/2 # shift breakpoint by deltax/2
  axs[2].fill_between([bp-deltax/2,bp+deltax/2],[fx(bp),fx(bp)],color='m',alpha=.4)



# plot titles
axs[0].set_title(f'Left rule: area={area_left:.3f}')
axs[1].set_title(f'Right rule: area={area_right:.3f}')
axs[2].set_title(f'Midpoint rule: area={area_midpoint:.3f}')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Demonstrate convergence with shrinking $\Delta$x

# In [ ]
Ns = np.arange(5,191,5)
areas = np.zeros((len(Ns),3))

# loop over discretization N's
for i,n in enumerate(Ns):
  deltax = (b-a)/n
  breakPoints = np.linspace(a,b,n+1)

  areas[i,0] = np.sum(fx( breakPoints[:-1] )) * deltax
  areas[i,1] = np.sum(fx( breakPoints[1:] )) * deltax
  areas[i,2] = np.sum(fx( (breakPoints[:-1]+breakPoints[1:])/2 )) * deltax



plt.figure(figsize=(10,5))
plt.plot(Ns,areas,linewidth=3)
plt.axhline(area_analytic,color='k',linestyle='--',linewidth=3)
plt.legend(['Left','Right','Midpoint','Analytic'])
plt.xlim(Ns[[0,-1]])
plt.xlabel('Number of bins')
plt.ylabel('Area (a.u.)')
plt.show()

# In [ ]

