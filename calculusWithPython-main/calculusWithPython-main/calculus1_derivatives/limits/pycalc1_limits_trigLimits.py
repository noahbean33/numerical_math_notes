# -*- coding: utf-8 -*-
# Auto-generated from 'pycalc1_limits_trigLimits.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master calculus 1 using Python: derivatives and applications
# ## SECTION: Limits
# ### LECTURE: CodeChallenge: Trig limits in sympy
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
# # Exercise 1: sin(theta)/theta

# In [ ]
# create the expression object
theta = sym.symbols('theta')
fx = sym.sin(theta) / theta

# visualization in sympy
sym.plot(fx,(theta,-100,100), adaptive=False, nb_of_points=1000,
         title=r'$f(\theta) = %s$'%sym.latex(fx),ylabel=None);

# In [ ]
# specify and plot g(t), f(t), and h(t)
gx = -1/theta
hx =  1/theta

sym.plot(fx,gx,hx,(theta,-100,100),ylim=[-1,1],
          adaptive=False, nb_of_points=1000);

# In [ ]
# now lambdify and plot in matplotlib
gFun = sym.lambdify(theta,gx)
fFun = sym.lambdify(theta,fx)
hFun = sym.lambdify(theta,hx)

xx = np.linspace(-100,100,10001) # why warnings with mod(N,2)=1?

plt.plot(xx,gFun(xx),'r',label='g($\\theta$)')
plt.plot(xx,fFun(xx),'k',label='f($\\theta$)')
plt.plot(xx,hFun(xx),'b',label='h($\\theta$)')

plt.ylim([-1,1])
plt.xlim(xx[[1,-1]])
plt.legend()
plt.grid()
plt.show()

# In [ ]
# plot the signs of the functions
plt.plot(xx,np.sign(fFun(xx)),'k',label='sign(f($\\theta$))')
plt.plot(xx,np.sign(gFun(xx))*1.1,'r',label='sign(g($\\theta$))',linewidth=3)
plt.plot(xx,np.sign(hFun(xx))*1.1,'b',label='sign(h($\\theta$))',linewidth=3)

plt.ylim([-1.2,1.2])
plt.yticks([-1.05,1.05],['neg','pos'])
plt.xlim(xx[[1,-1]])
plt.legend()
plt.show()

# In [ ]
# print out limits

print('\nLimit of g(t) as t approaches infinity')
print( sym.limit(gx,theta,sym.oo,dir='-') )

print('\nLimit of f(t) as t approaches infinity')
print( sym.limit(fx,theta,sym.oo,dir='-') )

print('\nLimit of h(t) as t approaches infinity')
print( sym.limit(hx,theta,sym.oo,dir='-') )

# In [ ]

# %% [markdown]
# # Exercise 2: tan(theta)/theta

# In [ ]
# functions in sympy
fx0 = sym.tan(theta) / theta
fx1 = sym.sin(theta) / theta
fx2 = 1 / sym.cos(theta)

# In [ ]
# now lambdify and plot in matplotlib
f0Fun = sym.lambdify(theta,fx0)
f1Fun = sym.lambdify(theta,fx1)
f2Fun = sym.lambdify(theta,fx2)

xx = np.linspace(-np.pi,np.pi,1001)
plt.plot(xx,f0Fun(xx),'r',label=r'$f_0=tan(\theta)/\theta$')
plt.plot(xx,f1Fun(xx),'k',label=r'$f_1=sin(\theta)/\theta$')
plt.plot(xx,f2Fun(xx),'b',label=r'$f_2=1/cos(\theta$)')
plt.plot(xx[::20],f1Fun(xx[::20])*f2Fun(xx[::20]),'ro',label=r'$f_1\times f_2$')

plt.ylim([-20,20])
plt.xlim(xx[[1,-1]])
plt.legend()
plt.grid()
plt.show()

# In [ ]
# print out values

print('\nValue of f0(t) at t=0:')
print( fx0.subs(theta,0) )

print('\nValue of f1(t) at t=0:')
print( fx1.subs(theta,0) )

print('\nValue of f2(t) at t=0:')
print( fx2.subs(theta,0) )

# In [ ]
# print out limits

print('\nLimit of f0(t) as t approaches zero:')
print( sym.limit(fx0,theta,0,dir='+-') )

print('\nLimit of f1(t) as t approaches zero:')
print( sym.limit(fx1,theta,0,dir='+-') )

print('\nLimit of f2(t) as t approaches zero:')
print( sym.limit(fx2,theta,0,dir='+-') )

# In [ ]

# %% [markdown]
# # Exercise 3: Some crazy weirdo function

# In [ ]
# create each part
parts = [theta**2 , sym.exp(-theta**2) , sym.log(theta**2) , sym.sin(theta)]

# compute f(theta) as their product
expr = 1
for p in parts:
  expr *= p

expr

# In [ ]
# sympy plotting
sym.plot(expr,(theta,-2*sym.pi,2*sym.pi),
         title=r'$f(\theta) = %s$'%sym.latex(expr),ylabel=None);

# In [ ]
# print out function value and limit at theta=0
print('\nValue of f(t) at t=0:')
print( expr.subs(theta,0) )

print('\nLimit of f(t) as t approaches zero:')
print( sym.limit(expr,theta,0,dir='+-') )

# In [ ]
print('Domain of function:')
sym.calculus.util.continuous_domain(expr,theta,sym.Reals)

# In [ ]
# list limit of each part, and compute their product
c = 1
for p in parts:
  print(f'Limit of {p} as theta->0 is {sym.limit(p,theta,0)}')
  c *= sym.limit(p,theta,0,dir='+-')

print(f'\n\nProduct of all limits is {c}')

# In [ ]

