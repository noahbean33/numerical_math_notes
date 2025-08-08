# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_integrate_CClinearity.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integrating functions
# ### LECTURE: CodeChallenge: Explore linearity
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
# # Exercise 1: Antiderivatives and scalar multiplication

# In [ ]
x,C = sym.symbols('x,C')

# create a function and scalar
f = x**2
s = sym.pi

# print
display(Math('f(x) = %s' %sym.latex(s*f)))

# In [ ]
# integrate the function with scalar inside or outside integrand
fs_int = sym.integrate(s*f,x) + C
sf_int = s * (sym.integrate(f,x) + C)

# print both
display(Math('\int %s dx = %s' %(sym.latex(f*s),sym.latex(fs_int))))
print('')
display(Math('%s\int %s dx = %s' %(sym.latex(s),sym.latex(f),sym.latex(sym.expand(sf_int)))))

# In [ ]

# %% [markdown]
# # Exercise 2: Definite integral with scalar multiplication

# In [ ]
# limits
a = 0
b = 3*sym.pi

# compute the definite integral with scalar inside or outside integrand
fs_defint = sym.integrate(f*s,(x,a,b))
sf_defint = s * sym.integrate(f,(x,a,b))

# print both results
display(Math('%s = %s' %(sym.latex(sym.Integral(f*s,(x,a,b))),sym.latex(fs_defint))))
display(Math('%s%s = %s' %(sym.latex(s),sym.latex(sym.Integral(f,(x,a,b))),sym.latex(sf_defint))))

# In [ ]

# %% [markdown]
# # Exercise 3: Geometry of scalar-multiplication linearity

# In [ ]
f_l(xx)

# In [ ]
# lambdify the function
f_l  = sym.lambdify(x,f)
fs_l = sym.lambdify(x,f*s)

# get x-axis grids for visualization (function and integration area)
xx = np.linspace(float(a)-np.pi/2,float(b)+np.pi/2,101)
x2integrate = np.linspace(float(a),float(b),101)

# create x-axis ticks and labels in radians
numTicks = int((b-a) / (sym.pi/2))
xtick_labels = []
xtick_vals = []
for i in range(numTicks):
  xtick_vals.append(a+i*(np.pi/2))
  xtick_labels.append('$%s$'%sym.latex(a+i*(sym.pi/2)))


# create the figure
_,ax = plt.subplots(1,figsize=(8,6))

# and plot all the lovely thingies
ax.plot(xx,f_l(xx),label=r'$f(x) = %s$'%sym.latex(f),color='r',linewidth=1)
ax.fill_between(x2integrate,f_l(x2integrate),alpha=.2,color='m')

ax.plot(xx,np.ones(len(xx))*fs_l(xx),label=r'$f(x) = %s$'%sym.latex(s*f),color='b',linewidth=1)
ax.fill_between(x2integrate,fs_l(x2integrate),color='none', hatch='X', edgecolor='b',zorder=-10)

ax.axhline(0,color='k',linestyle='--',zorder=-4,linewidth=.5)
ax.set(xlabel='x',ylabel='y',xlim=xx[[0,-1]],xticks=xtick_vals,xticklabels=xtick_labels)
ax.legend()

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 4: Functions with summed terms

# In [ ]
# create the two functions and their sum
f = x**2
g = 10*sym.sin(x)
h = f+g

# print
display(Math('f(x) = %s' %sym.latex(f)))
display(Math('g(x) = %s' %sym.latex(g)))
display(Math('h(x) = %s' %sym.latex(h)))

# In [ ]
# integrate all three functions
fi = sym.integrate(f)
gi = sym.integrate(g)
hi = sym.integrate(h)


# show the results
display(Math('f:\quad %s = %s' %(sym.latex(sym.Integral(f)),sym.latex(fi+C))))
display(Math('g:\quad %s = %s' %(sym.latex(sym.Integral(g)),sym.latex(gi+C))))
display(Math('f+g:\quad %s = %s' %(sym.latex(sym.Integral(f+g)),sym.latex(fi+gi+C))))
print('')
display(Math('h:\quad %s = %s' %(sym.latex(sym.Integral(h)),sym.latex(hi+C))))

# In [ ]

# %% [markdown]
# # Exercise 5: Definite integrals of summed functions

# In [ ]
# limits
a = 0
b = 3*sym.pi

# integrate
f_defint = sym.integrate(f,(x,a,b))
g_defint = sym.integrate(g,(x,a,b))
h_defint = sym.integrate(h,(x,a,b))

display(Math('\int_{%s}^{%s}%s \,dx = %g' %(sym.latex(a),sym.latex(b),sym.latex(f),f_defint)))
print('')
display(Math('\int_{%s}^{%s}%s \,dx = %g' %(sym.latex(a),sym.latex(b),sym.latex(g),g_defint)))
print('')
display(Math('\int_{%s}^{%s}%s \,dx = %g' %(sym.latex(a),sym.latex(b),sym.latex(h),h_defint)))

# In [ ]

# %% [markdown]
# # Exercise 6: Visualize the summed functions

# In [ ]
# lambdify all functions
f_l = sym.lambdify(x,f)
g_l = sym.lambdify(x,g)
h_l = sym.lambdify(x,h)

# get x-axis grids for visualization (function and integration area)
xx = np.linspace(float(a)-np.pi/2,float(b)+np.pi/2,101)
x2integrate = np.linspace(float(a),float(b),101)

# plot all the lovely things
_,axs = plt.subplots(1,3,figsize=(15,4))

# f(x)
axs[0].plot(xx,f_l(xx),label='f(x)',color='r')
axs[0].axhline(0,color='k',linestyle='--',zorder=-4,linewidth=.5)
axs[0].fill_between(x2integrate,f_l(x2integrate),alpha=.2,color='r')
axs[0].set(xlabel='x',ylabel='f(x)',xlim=xx[[0,-1]])
axs[0].set_title(r'$\int f \,dx = %.2f$' %f_defint,color='r')

# g(x)
axs[1].plot(xx,g_l(xx),label='g(x)',color='b')
axs[1].axhline(0,color='k',linestyle='--',zorder=-4,linewidth=.5)
axs[1].fill_between(x2integrate,g_l(x2integrate),alpha=.2,color='b')
axs[1].set(xlabel='x',ylabel='g(x)',xlim=xx[[0,-1]])
axs[1].set_title(r'$\int g \,dx = %.2f$' %g_defint,color='b')

# h(x) = f+g
axs[2].plot(xx,h_l(xx),label='h(x)',color='m')
axs[2].axhline(0,color='k',linestyle='--',zorder=-4,linewidth=.5)
axs[2].fill_between(x2integrate,h_l(x2integrate),alpha=.2,color='m')
axs[2].set(xlabel='x',ylabel='h(x)',xlim=xx[[0,-1]])
axs[2].set_title(r'$\int h \,dx = %.2f$' %h_defint,color='m')

plt.tight_layout()
plt.show()

# In [ ]
# all functions in the same graph with a patch for h(x)

_,ax = plt.subplots(1,figsize=(8,6))

ax.plot(xx,f_l(xx),label='f(x)',color='r',linewidth=1)
ax.plot(xx,g_l(xx),label='g(x)',color='b',linewidth=1)
ax.plot(xx,h_l(xx),label='h(x)',color='m',linewidth=2)
ax.fill_between(x2integrate,h_l(x2integrate),alpha=.2,color='m')

ax.axhline(0,color='k',linestyle='--',zorder=-4,linewidth=.5)
ax.set(xlabel='x',ylabel='y',xlim=xx[[0,-1]])
ax.legend()

plt.tight_layout()
plt.show()

# In [ ]

