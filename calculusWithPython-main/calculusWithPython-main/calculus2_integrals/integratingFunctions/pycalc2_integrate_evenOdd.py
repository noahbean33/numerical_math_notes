# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_integrate_evenOdd.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integrating functions
# ### LECTURE: Integrating even and odd functions
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
# # Determine whether a function is even or odd

# In [ ]
x = sym.symbols('x')

# some math functions
f1 = x**2
f2 = x**3
f3 = x**2 + x**3


# a python function
def isFunEvenOrOdd(fx):

  # even if f(x) = f(-x)
  is_even = sym.simplify(fx.subs(x,-x) - fx) == 0

  # odd if f(-x) = -f(x)
  is_odd = sym.simplify(fx.subs(x,-x) + fx) == 0

  # return a string result
  if is_even:
    result = 'even'
  elif is_odd:
    result = 'odd'
  else:
    result = 'neither even nor odd'

  return result


for ff in (f1,f2,f3):
  display(Math('f(x) = %s \\text{ is } \\text{%s}.' %(sym.latex(ff),isFunEvenOrOdd(ff))))

# In [ ]

# %% [markdown]
# # Demonstrate integration shortcut

# In [ ]
# a function
fx = -4*x**5 + x**3 - 10*x
fx = sym.cos(3*x) + (x-0)**2 - 3

# integration bounds
a = -3
b = np.abs(a)

# compute integrals
int_a2b = sym.integrate(fx,(x,a,b))
int_02b = sym.integrate(fx,(x,0,b))

# print some results
display(Math('\\text{This function is %s}.' %isFunEvenOrOdd(fx)))
display(Math('%s = %g' %(sym.latex(sym.Integral(fx,(x,a,b))),int_a2b)))
display(Math('%s = %g' %(sym.latex(sym.Integral(fx,(x,0,b))),int_02b)))
display(Math('2%s = %g' %(sym.latex(sym.Integral(fx,(x,0,b))),2*int_02b)))

# In [ ]

# %% [markdown]
# # Visualize the symmetry

# In [ ]
# define x-axis grid and lambdify the function
xx = np.linspace(a-.25,b+.25,117)
fun_l = sym.lambdify(x,fx)


plt.figure(figsize=(12,5))

# plot the function graph, the y=0 line, and lines for the integration bounds
plt.plot(xx,fun_l(xx),'k',linewidth=2,label=r'$f(x) = %s$' %sym.latex(fx))
plt.axhline(0,color='lightgray',linestyle='--',zorder=-3)
plt.axvline(0,color='lightgray',linestyle='--',zorder=-3)
plt.axvline(a,color='lightblue',linestyle='--',zorder=-3)
plt.axvline(b,color='lightblue',linestyle='--',zorder=-3)

plt.plot(a,fun_l(a),'o',markersize=10,label=r'$f(%g)=%g$' %(a,fun_l(a)))
plt.plot(b,fun_l(b),'o',markersize=10,label=r'$f(%g)=%g$' %(a,fun_l(b)))

# find x values for this segment and draw a patch
xxSegment = xx[(xx>a) & (xx<0)]
plt.fill_between(xxSegment,fun_l(xxSegment),color='b',alpha=.2)

xxSegment = xx[(xx>=0) & (xx<b)]
plt.fill_between(xxSegment,fun_l(xxSegment),color='m',alpha=.2)


plt.gca().set(xlim=xx[[0,-1]],ylim=[np.min(fun_l(xx))-1,np.max(fun_l(xx))+1],
              xlabel='x',ylabel='y = f(x)')
plt.legend()
plt.show()

# In [ ]

