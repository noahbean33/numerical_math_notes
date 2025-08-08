# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_differentiation_GlobalLocalSlopes.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation fundamentals
# ### LECTURE: CodeChallenge: Global and local slopes
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc1_x/?couponCode=202307

# In [ ]

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# better image resolution
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# In [ ]

# %% [markdown]
# # Exercise 1: global slope

# In [ ]
# define points as XY pairs
p1 = [-1,1]
p2 = [3,6]

# compute slope
m = (p2[1]-p1[1]) / (p2[0]-p1[0])

# plot the dots and a line between them
plt.plot([p1[0],p2[0]],[p1[1],p2[1]],'k')
plt.plot(p1[0],p1[1],'ro',label='p1')
plt.plot(p2[0],p2[1],'bs',label='p2')

# put the slope in the title
plt.title(f'The slope of the line is m={m}')

# make the plot look a bit nicer
plt.plot([-2,7],[0,0],'k--',linewidth=.3)
plt.plot([0,0],[-2,7],'k--',linewidth=.3)

plt.xlim([-2,7])
plt.ylim([-2,7])
plt.legend()
plt.gca().set_aspect('equal')
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: local slopes

# In [ ]
# define function
N = 5
x = np.linspace(-1,5,N)
y = x**2


_,axs = plt.subplots(2,1,figsize=(5,8))
axs[0].plot(x,y,'o-')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y=f(x)')
axs[0].set_title('Function')


# compute and plot the slopes
m = np.zeros(N-1)
for i in range(1,N):
  m[i-1]  = y[i]-y[i-1]
  m[i-1] /= x[i]-x[i-1]
  axs[1].plot([x[i-1],x[i]],[m[i-1],m[i-1]],linewidth=3)

axs[1].set_title('Segment slopes')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Local slope (m)')

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Global and average of local slopes

# In [ ]
globalSlope = (y[-1]-y[0]) / (x[-1]-x[0])
aveLocalSlopes = np.mean(m)

print('       Global: %g' %globalSlope)
print('Ave of locals: %g' %aveLocalSlopes)

# In [ ]

