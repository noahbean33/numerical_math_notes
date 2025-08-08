# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc2_techniques_CCfunWithFunctions.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 2 using Python: integration, intuition, code
# ## SECTION: Integration techniques
# ### LECTURE: CodeChallenge: Fun with functions
# #### TEACHER: Mike X Cohen, sincxpress.com
# ##### COURSE URL: udemy.com/course/pycalc2_x/?couponCode=202505

# In [ ]
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from IPython.display import display,Math
import scipy.integrate as spi


# adjust matplotlib defaults to personal preferences
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.rcParams.update({'font.size':14,             # font size
                     'axes.spines.right':False,  # remove axis bounding box
                     'axes.spines.top':False,    # remove axis bounding box
                     })

# In [ ]

# %% [markdown]
# # Exercise 1: Areas of a parameterized trig function

# In [ ]
x = np.linspace(-2,4,402)

As = np.linspace(0,1,31)
Bs = np.linspace(0,2,15)

Areas = np.zeros((len(As),len(Bs)))

fig,axs = plt.subplots(1,2,figsize=(14,5))

for ai,a in enumerate(As):
  for bi,b in enumerate(Bs):

    # create the function (unique for this a/b parameter pair)
    fx = lambda xx : np.abs(xx)*np.cos((a+xx)**2)+b

    # plot it
    axs[0].plot(x,fx(x),linewidth=a,color=[1-b/2,np.min((1,np.sqrt(b/2))),1-b/2])

    # calculate area and store
    Areas[ai,bi] = spi.quad(fx,x[0],x[-1])[0]


# draw the areas
im = axs[1].imshow(Areas,extent=[Bs[0],Bs[-1],As[-1],As[0]],cmap='turbo')
axs[1].set(xlabel='"b" parameter',ylabel='"a" parameter',title=r'$\int_{-2}^{4}f(x,a,b)\,dx$')
fig.colorbar(im,ax=axs[1],shrink=.6)

# final touch-ups
axs[0].axis('off')
plt.tight_layout()
plt.show()

# In [ ]

# %% [markdown]
# # Exercise 2: The complexity of rationality

# In [ ]
# symbolic variable (using 't' here bc I used 'x' above)
t = sym.symbols('t')

# create the function
ft = (t**2+4) / (3*t**3-4*t**2-4*t)

# take its 1st and 2nd integral
antideriv = sym.integrate(ft)
antideriv2 = sym.integrate(antideriv)

# show the equations
display(Math('%s = %s+C' %(sym.latex(sym.Integral(ft)),sym.latex(antideriv))))
print('')
display(Math('%s = %s+C' %(sym.latex(sym.Integral(antideriv)),sym.latex(antideriv2))))

# In [ ]

# discretize the function
tt = np.linspace(-np.pi,2*np.pi,402)
ft_l = sym.lambdify(t,ft)
if_l = sym.lambdify(t,antideriv)
if_l2 = sym.lambdify(t,antideriv2)

# and make some lovely plots :)
_,axs = plt.subplots(1,2,figsize=(14,5))

axs[0].plot(tt,ft_l(tt),'k',label='f(t)')
axs[0].plot(tt,np.real(if_l(tt.astype(complex))),'b',label=r'$re\left(\int\right)$')
axs[0].plot(tt,np.imag(if_l(tt.astype(complex))),'b--',label=r'$im\left(\int\right)$')
axs[0].plot(tt,np.real(if_l2(tt.astype(complex))),'r',label=r'$re\left(\int\int\right)$')
axs[0].plot(tt,np.imag(if_l2(tt.astype(complex))),'r--',label=r'$im\left(\int\int\right)$')
axs[0].set(xlim=tt[[0,-1]],ylim=[-10,10],xlabel='t',ylabel='y=f(t) or F(t)',title='Function and integrals in x-space')
axs[0].legend(ncol=2,fontsize=12)

axs[1].plot(np.real(if_l(tt.astype(complex))),np.imag(if_l(tt.astype(complex))),'r.',linewidth=2)
axs[1].plot(np.real(if_l2(tt.astype(complex))),np.imag(if_l2(tt.astype(complex))),'b.',linewidth=2)
axs[1].set(xlabel='Real part',ylabel='Imaginary part',title='Definite integrals in complex plane')

plt.tight_layout()
plt.show()

# In [ ]

