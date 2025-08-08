# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_geometry_parametric.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Applications in geometry
# ### LECTURE: Parametric curves
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
t = np.linspace(.0001,np.pi,323)

x1 = np.cos(t)
x2 = np.cos(np.log(abs(t)))
y  = np.sin(t**2)

_,axs = plt.subplots(1,3,figsize=(12,3.5))
axs[0].plot(x1,y)
axs[0].text(0,-.4,'$x=\cos(t)$\n' + '$y=\sin(t^2)$',fontsize=16)
axs[0].set(xlabel='$x(t)$',ylabel='$y(t)$',title='Parametric curve 1')

axs[1].plot(x2,y)
axs[1].text(-.7,-.2,'$x=\cos(\ln|t|)$\n' + '$y=\sin(t^2)$',fontsize=16)
axs[1].set(xlabel='$x(t)$',ylabel='$y(t)$',title='Parametric curve 2')

axs[2].plot(t,x2,label=r'$x=\cos(2t)$')
axs[2].plot(t,y,label=r'$y=\sin(t^2)$')
axs[2].legend()
axs[2].set(xlim=t[[0,-1]],xlabel=r'$t$',ylabel=r'$x(t)$ or $y(t)$',title='Explicit curves')


plt.tight_layout()
plt.show()

# In [ ]

# In [ ]
# organized as a list of dictionaries

functions = [
  { # panel A
    't': np.linspace(-2*np.pi, 2*np.pi, 301),
    'x': lambda t: np.sin(2*t) + np.sin(t),
    'y': lambda t: 2*np.sin(3*t)
  },
  { # panel B
    't': np.linspace(0, 2*np.pi, 101),
    'x': lambda t: 3*np.cos(t) + np.cos(3*t),
    'y': lambda t: 3*np.sin(t) - np.sin(4*t)
  },
  { # panel C
    't': np.linspace(0, 12*np.pi, 1001),
    'x': lambda t: np.sin(t)*(np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5),
    'y': lambda t: np.cos(t)*(np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5)
  },
  { # panel D
    't': np.linspace(0, 12*np.pi, 1001),
    'x': lambda t: np.sqrt(3)*np.cos(2*t) - np.cos(10*t)*np.sin(20*t),
    'y': lambda t: -np.sqrt(2)*np.sin(2*t) - np.sin(10*t)*np.sin(20*t)
  },
  { # panel E
    't': np.linspace(0, 2*np.pi, 5001),
    'x': lambda t: np.cos(t) + np.cos(52*t)/2 + np.sin(25*t)/3,
    'y': lambda t: np.sin(t) + np.sin(52*t)/2 + np.cos(25*t)/3
  },
  { # panel F
    't': np.linspace(-2*np.pi, 2*np.pi, 10001),
    'x': lambda t: 2.5*np.sin(np.sin(-5*t)) * np.cos(9.844*t)**2,
    'y': lambda t: 2.5*np.sin(-5*t)**2 * 2**(np.cos(np.cos(9.844*t)))
  }
]

# Loop over functions in groups of 3
_,axs = plt.subplots(2,3,figsize=(12,6))
axs = axs.flatten()
titles = 'ABCDEF'

for i in range(len(functions)):

  # extract the data
  t = functions[i]['t']
  x = functions[i]['x'](t)
  y = functions[i]['y'](t)

  # and plot
  axs[i].plot(x,y,'k')
  axs[i].axis('off')
  axs[i].set_title(fr'Function ($\bf{{{titles[i]}}}$)')


plt.tight_layout()
plt.show()

# In [ ]

# In [ ]
# with more interesting colors

# extract the data
t = functions[5]['t']
x = functions[5]['x'](t)
y = functions[5]['y'](t)

plt.figure(figsize=(4,4))
plt.scatter(x,y,s=5,c=t,cmap='Purples')
plt.axis('off')

plt.tight_layout()
plt.show()

# In [ ]

