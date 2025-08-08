# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geoApprox_CClebesgue.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Geometric approximations
# ### LECTURE: CodeChallenge: Lebesgue in Blefuscu
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
# # Exercise 1: Riemann approximation

# In [ ]
# function for the function
def fx(u):
  return u**2 + np.cos(2*np.pi*u)/5

# In [ ]
# create deltax
n = 12
a = 0
b = 1
deltax = (b-a)/n

breakPoints = [ a+deltax*i for i in range(n+1) ]


# plot the function
_,axs = plt.subplots(1,figsize=(10,6))

# plot the function
xx = np.linspace(a-.1,b+.1,301)
axs.plot(xx,fx(xx),'r',markersize=10,markerfacecolor=[.7,.3,.9])

# initialize area
riemann_approx = 0

# plot rectangles
for i in range(n):

  # compute the function value at midpoint
  bp = breakPoints[i] + deltax/2 # shift breakpoint by deltax/2
  y  = fx(bp)

  # draw the rectangle
  faCo = np.array([.7,.3,1]) * (1-i/n)
  axs.fill_between([breakPoints[i],breakPoints[i+1]],[y,y],edgecolor="deeppink",facecolor=faCo)

  # sum the area
  riemann_approx += y * deltax

# set the labels (*after* the for-loop)
axs.set(xlabel='x',ylabel=r'$y = x^2+\cos(2\pi x)/5$')
axs.set_title(r'Net area = %.3f $\Delta$x=%g' %(riemann_approx,deltax),wrap=True)


# finalize
plt.axhline(0,color='gray',linestyle='--')
plt.axvline(a,color='gray',linestyle='--')
plt.axvline(b,color='gray',linestyle='--')
plt.xlim(xx[[0,-1]])
plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: Lebesgue approximation

# In [ ]
# fine partitioning of the x-axis (domain)
domain_n = 1000
domainPoints = np.linspace(a,b,domain_n)
deltax = (b-a)/domain_n

# evaluate the function at those points
fx_values = fx(domainPoints)

# determine the range of the function in this domain
min_val, max_val = np.min(fx_values), np.max(fx_values)

# define boundaries for range of fx
yPartitions = np.linspace(min_val,max_val,n+1)
deltay = yPartitions[1]-yPartitions[0]

# initialize Lebesgue approximation
lebesgue_approx = 0




# plot the function
_,axs = plt.subplots(1,figsize=(10,6))

# plot the function
xx = np.linspace(a-.1,b+.1,301)
axs.plot(xx,fx(xx),'r',markersize=10,markerfacecolor=[.7,.3,.9])


for i in range(n):

  # find points where the function is within the current partition
  in_partition = (fx_values >= yPartitions[i]) & (fx_values <= yPartitions[i+1])

  # measure the "size" of that set
  measure = np.sum(in_partition) * deltax

  # The average function value on this set
  average_value = (yPartitions[i] + yPartitions[i+1]) / 2

  # sum this set to the integral approximation
  lebesgue_approx += average_value * measure


  ### finished the calculations; the code below is just visualization

  # find the contiguous groups in in_partition
  in_partition_diff = np.diff(in_partition.astype(int))
  group_starts = np.where(in_partition_diff == 1)[0] + 1  # start points of groups
  group_ends = np.where(in_partition_diff == -1)[0]       # end points of groups

  # in case a group starts/ends at the integration bounds
  if in_partition[0]:
    group_starts = np.insert(group_starts, 0, 0)
  if in_partition[-1]:
    group_ends = np.append(group_ends, len(in_partition) - 1)


  # loop over groups and draw rectangles for each
  for start, end in zip(group_starts, group_ends):

    # visualization option "a"
    x1, x2 = domainPoints[start], domainPoints[end]
    y1, y2 = yPartitions[i], yPartitions[i+1]

    # visualization option "b"
    x1, x2 = domainPoints[start], domainPoints[end]
    y1, y2 = 0, yPartitions[i+1]

    # visualization option "c"
    x1 = a if fx(domainPoints[end])<fx(domainPoints[start]) else domainPoints[start]
    x2 = b if fx(domainPoints[end])>fx(domainPoints[start]) else domainPoints[end]
    y1, y2 = yPartitions[i],yPartitions[i+1]

    # draw the patch
    faCo = np.array([.7, .3, 1]) * (1-i/n) # set face color
    axs.fill_between([x1, x2], y1, y2, color=faCo, edgecolor="deeppink")



# set the labels (after the for-loop)
axs.set(xlabel='x',ylabel=r'$y = x^2+\cos(2\pi x)/5$')
axs.set_title(r'Net area = %.3f $\Delta$y=%g' %(lebesgue_approx,deltay),wrap=True)


# finalize
plt.axhline(0,color='gray',linestyle='--')
plt.axvline(a,color='gray',linestyle='--')
plt.axvline(b,color='gray',linestyle='--')
plt.xlim(xx[[0,-1]])
plt.yticks(yPartitions)
plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Compare Riemann, Lebesgue, and analytical

# In [ ]
# Riemann
def riemann():
  deltax = (b-a)/n
  breakPoints = [ a+deltax*i for i in range(n+1) ]
  riemann_approx = 0
  for i in range(n):
    bp = breakPoints[i] + deltax/2
    riemann_approx += fx(bp) * deltax
  return riemann_approx



# Lebesgue
def lebesgue():
  domain_n = 1000
  domainPoints = np.linspace(a,b,domain_n)
  deltax = 1/domain_n*(b-a)

  # evaluate the function at those points
  fx_values = fx(domainPoints)

  # determine the range of the function in this domain
  min_val, max_val = np.min(fx_values), np.max(fx_values)

  # define boundaries for range of fx
  yPartitions = np.linspace(min_val,max_val,n+1)
  deltay = yPartitions[1]-yPartitions[0]

  # initialize Lebesgue approximation
  lebesgue_approx = 0

  for i in range(n):
    in_partition = (fx_values >= yPartitions[i]) & (fx_values < yPartitions[i+1])
    measure = np.sum(in_partition) * deltax
    average_value = (yPartitions[i] + yPartitions[i+1]) / 2
    lebesgue_approx += average_value * measure

  return lebesgue_approx

# In [ ]
# number of partitions (same for Riemann and Lebesgue)
n = 12

# run the functions
R = riemann()
L = lebesgue()

# calculate the true integral using sympy
from sympy.abc import x
fx_s = x**2 + sym.cos(2*sym.pi*x)/5
A = sym.integrate(fx_s,(x,a,b))

# report the results!
print(f'Riemann:  {R:.6f}')
print(f'Lebesgue: {L:.6f}')
print(f'Analytic: {A:.6f}')

# In [ ]
# range of discretizations
Ns = np.arange(4,41)

# initialize results
riemann_results = np.zeros(len(Ns))
lebesgue_results = np.zeros(len(Ns))

# run the experiment!
for i,n in enumerate(Ns):
  riemann_results[i] = riemann()
  lebesgue_results[i] = lebesgue()


# plot the results
_,ax = plt.subplots(1,figsize=(10,5))
ax.plot(Ns,riemann_results,'k',linewidth=2,label='Riemann')
ax.plot(Ns,lebesgue_results,'r',linewidth=2,label='Lebesgue')
ax.axhline(A,linestyle='--',linewidth=2,color='blue',label='Sympy')
ax.set(xlabel='Partitions',xlim=Ns[[0,-1]],ylabel='Integral')
ax.legend()
plt.show()

# In [ ]

