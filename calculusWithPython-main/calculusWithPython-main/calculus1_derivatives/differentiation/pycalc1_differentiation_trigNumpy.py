# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_differentiation_trigNumpy.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Differentiation fundamentals
# ### LECTURE: CodeChallenge: trig derivatives
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
# # Exercise 1: d/dx cos(x)

# In [ ]
dx = .01
x = np.arange(-1.5*np.pi,1.5*np.pi,step=dx)

# define the function and its difference (same as derivative?)
fx  = np.cos(x)
dfx = np.diff(fx) / dx


plt.figure(figsize=(8,5))
plt.plot(x,fx,label='cos(x)',linewidth=2)
plt.plot(x[:-1],dfx,label='diff(cos(x))',linewidth=2)
plt.plot(x[::20],-np.sin(x[::20]),'o',label='-sin(x)')

plt.legend()
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels=[r'$-\pi$',r'$-\pi/2$','0',r'$\pi/2$',r'$\pi$'])
plt.xlabel('Angle (rad.)')
plt.xlim(x[[0,-1]])
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: d/dx sin(x)

# In [ ]
dx = .01
x = np.arange(-1.5*np.pi,1.5*np.pi,step=dx)

# define the function and its difference (same as derivative?)
fx  = np.sin(x)
dfx = np.diff(fx) / dx


plt.figure(figsize=(8,5))
plt.plot(x,fx,label='sin(x)',linewidth=2)
plt.plot(x[:-1],dfx,label='diff(sin(x))',linewidth=2)
plt.plot(x[::20],np.cos(x[::20]),'o',label='cos(x)')

plt.legend()
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels=[r'$-\pi$',r'$-\pi/2$','0',r'$\pi/2$',r'$\pi$'])
plt.xlabel('Angle (rad.)')
plt.xlim(x[[0,-1]])
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 3: Cyclicity of trig derivatives

# In [ ]
# symbolic variable
x = sym.symbols('x')

# start with cos
f = sym.cos(x)

# cycle through
for i in range(4):
  print(f"({f})' = {sym.diff(f)}")
  f = sym.diff(f)

# In [ ]

# %% [markdown]
# # Exercise 4: cyclic inverse trig derivatives

# In [ ]
# start with acos
f = sym.acos(x)

# cycle through
for i in range(4):
  print(f"({f})' = {sym.diff(f)}")
  f = sym.diff(f)

# In [ ]

