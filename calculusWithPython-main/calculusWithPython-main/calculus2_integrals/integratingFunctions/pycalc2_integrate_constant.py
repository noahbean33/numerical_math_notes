# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_integrate_constant.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integrating functions
# ### LECTURE: Integration constants in Python
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
# # Setup the functions

# In [ ]
from sympy.abc import x

# define functions (same but for constant)
fx = x**2/10 + sym.sin(x) + 0
gx = x**2/10 + sym.sin(x) + 1
hx = x**2/10 + sym.sin(x) + 2

# their derivatives
fx_d = sym.diff(fx)
gx_d = sym.diff(gx)
hx_d = sym.diff(hx)

# integrals of the derivatives
fx_i = sym.integrate(fx_d)
gx_i = sym.integrate(gx_d)
hx_i = sym.integrate(hx_d)

# integrals of the functions
fx_if = sym.integrate(fx)
gx_if = sym.integrate(gx)
hx_if = sym.integrate(hx)

# quickie plot
sym.plot(fx,gx,hx,(x,-2*sym.pi,2*sym.pi));

# In [ ]

# %% [markdown]
# # Add a constant of integration

# In [ ]
# show the function and its integral
display(Math('f(x) = ' + sym.latex(fx))), print('')
display(Math('F(x) = ' + sym.latex(fx_if))), print('\n\n')

# add in a constant
C = sym.symbols('C')
fx_if_withC = fx_if + C
display(Math(sym.latex(sym.Integral(fx)) + ' = ' + sym.latex(fx_if_withC)))

# In [ ]
# substitute for the variables
val4x = 0
val4C = 9

# use a dictionary to substitute multiple variables
fx_if_withC.subs({x:val4x,C:val4C})

# In [ ]
# or can use multiple subs methods
fx_if_withC.subs(x,val4x).subs(C,val4C).evalf()

# In [ ]

# %% [markdown]
# # Visualizations

# In [ ]
# x-axis points to evaluate functions
xx = np.linspace(-2*np.pi,2*np.pi,75)

# note: lambdify is faster and more accurate, but list comprehension is also good (sometimes necessary)
fx_xeval = [fx.subs(x,i) for i in xx]
gx_xeval = [gx.subs(x,i) for i in xx]
hx_xeval = [hx.subs(x,i) for i in xx]



# and make the plts
_,axs = plt.subplots(2,2,figsize=(12,8))
axs[0,0].plot(xx,fx_xeval,linewidth=2,label=r'$f(x) = %s$' %sym.latex(fx))
axs[0,0].plot(xx,gx_xeval,linewidth=2,label=r'$g(x) = %s$' %sym.latex(gx))
axs[0,0].plot(xx,hx_xeval,linewidth=2,label=r'$h(x) = %s$' %sym.latex(hx))

axs[0,1].plot(xx,[fx_d.subs(x,i) for i in xx],'-',label="f'")
axs[0,1].plot(xx,[gx_d.subs(x,i) for i in xx],'o',label="g'")
axs[0,1].plot(xx,[hx_d.subs(x,i) for i in xx],'x',label="h'")

axs[1,0].plot(xx,[fx_i.subs(x,i) for i in xx],'-',label=r"$\int f' dx$") # antiderivatives of the functions
axs[1,0].plot(xx,[gx_i.subs(x,i) for i in xx],'o',label=r"$\int g' dx$")
axs[1,0].plot(xx,[hx_i.subs(x,i) for i in xx],'x',label=r"$\int h' dx$")

axs[1,1].plot(xx,[fx_if.subs(x,i) for i in xx],'-',label=r"$F(X) = \int f \,dx$") # integrals of functions
axs[1,1].plot(xx,[gx_if.subs(x,i) for i in xx],'o',label=r"$G(X) = \int g \,dx$")
axs[1,1].plot(xx,[hx_if.subs(x,i) for i in xx],'x',label=r"$H(X) = \int h \,dx$")


# axis adjustments
for a in axs.flatten():
  a.set(xlabel='x',ylabel='y',xlim=xx[[0,-1]])
  a.legend()

plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Visualization of why C is unnecessary for definite integrals

# In [ ]
# add various constants to the antiderivative
fx_if_0 = fx_if + 0
fx_if_1 = fx_if + 1
fx_if_2 = fx_if + 2

# integration limits
a,b = 0,2


# and plot!
plt.figure(figsize=(8,6))

# plot integrals with different constants
plt.plot(xx,[fx_if_0.subs(x,i) for i in xx],linewidth=2,label=r'F(x) + 0')
plt.plot(xx,[fx_if_1.subs(x,i) for i in xx],linewidth=2,label=r'F(x) + 1')
plt.plot(xx,[fx_if_2.subs(x,i) for i in xx],linewidth=2,label=r'F(x) + 2')

# dots on F(x) at bounds (definite integral is *change* in y-axis F(b)-F(a) )
plt.plot([a,b],[fx_if_0.subs(x,a),fx_if_0.subs(x,b)],'b-o',label=r'$\int_{%g}^{%g} f(x) \, dx$, C=0' %(a,b))
plt.plot([a,b],[fx_if_1.subs(x,a),fx_if_1.subs(x,b)],'r-o',label=r'$\int_{%g}^{%g} f(x) \, dx$, C=1' %(a,b))
plt.plot([a,b],[fx_if_2.subs(x,a),fx_if_2.subs(x,b)],'k-o',label=r'$\int_{%g}^{%g} f(x) \, dx$, C=2' %(a,b))

# final niceties
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim([-np.pi,np.pi])
plt.ylim([-1.5,4])
plt.legend(loc='upper left',fontsize=12)
plt.show()

# In [ ]

