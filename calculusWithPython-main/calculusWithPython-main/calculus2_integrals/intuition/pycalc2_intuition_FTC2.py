# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_intuition_FTC2.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration and applications
# ## SECTION: Intuition for integration
# ### LECTURE: The fundamental theorem of calculus, Part 2
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202506

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
# # FTC2: the definite integral is the difference at the bounds

# In [ ]
# create symbolic variables
x = sym.symbols('x')

# create a function
fx = (x-1)**2

# define two boundaries
a = -.5
b = sym.pi

# integrals
int_def   = sym.integrate(fx,(x,a,b))
int_indef = sym.integrate(fx,x)

# indefinite integral evaluated at each boundary
int_a = int_indef.subs(x,a)
int_b = int_indef.subs(x,b)

# show some results
display(Math('f(x) = %s' %sym.latex(fx))), print('')
display(Math('%s = %s' %(sym.latex(sym.Integral(fx,x)),sym.latex(int_indef))))

# In [ ]
# print the results
display(Math('F(%s) = %s' %(sym.latex(a),sym.latex(int_a))))
display(Math('F(%s) = %s' %(sym.latex(b),sym.latex(int_b))))
display(Math('F(%s)-F(%s) = %g' %(sym.latex(b),sym.latex(a),int_b-int_a)))
print('')
display(Math('\\int_{%s}^{%s} %s dx = %g' %(sym.latex(a),sym.latex(b),sym.latex(fx),int_def)))

# In [ ]

# %% [markdown]
# # Visualization

# In [ ]
# lambdify the function and its integral
fx_l  = sym.lambdify(x,fx)
int_l = sym.lambdify(x,int_indef)

# evaluate the integral at the limits
int_at_a = int_l(float(a))
int_at_b = int_l(float(b))


# define x-axis grid
xx = np.linspace(-1,4,45)

# and plot
plt.figure(figsize=(10,5))
plt.plot(xx,fx_l(xx),'k',linewidth=2,label=r'$f(x) = %s$'%sym.latex(fx))
plt.plot(xx,int_l(xx),'g',linewidth=2,label=r'$F(x) = %s$'%sym.latex(int_indef))

# integration limits and their "raw" integrals
plt.axvline(a,linestyle=':',color='r',label=r'$x_a$ = %.2f'%a)
plt.axvline(b,linestyle=':',color='b',label=r'$x_b$ = %.2f'%b)

plt.text(a+.03,fx_l(a)*1.4,r'$F(%.2f) = %.2f$' %(a,int_at_a),rotation=90)
plt.text(b,fx_l(b)*1.06,r'$F(%.2f) = %.2f$' %(b,int_at_b),rotation=90,ha='right')


# area between function curve and x-axis
x4area = np.linspace(float(a),float(b),99)
plt.fill_between(x4area,fx_l(x4area),label=f'Area = {int_at_b-int_at_a:.2f}',
                 color='m',edgecolor=None,alpha=.2)

# other plotting niceties
plt.axhline(0,linestyle='--',color=(.7,.7,.7),zorder=-10)

plt.gca().set(xlabel='x',ylabel='y=f(x)',xlim=xx[[0,-1]])
plt.legend()
plt.show()

# In [ ]

