# -*- coding: utf-8 -*-
# Auto-generated from 'stats_dataJourney.ipynb' on 2025-08-08T15:22:56
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# # COURSE: Master statistics and machine learning: Intuition, Math, code
# ##### COURSE URL: udemy.com/course/statsml_x/?couponCode=202304
# ## SECTION: A real-world data journey
# #### TEACHER: Mike X Cohen, sincxpress.com

# In [ ]
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA
import pandas as pd

# In [ ]
# data urls

marriage_url = 'https://www.cdc.gov/nchs/data/dvs/state-marriage-rates-90-95-99-19.xlsx'
divorce_url  = 'https://www.cdc.gov/nchs/data/dvs/state-divorce-rates-90-95-99-19.xlsx'

# %% [markdown]
# # Import the marriage data

# In [ ]
data = pd.read_excel(marriage_url,header=5)
data

# In [ ]
# remove irrelevant rows
data.drop([0,52,53,54,55,56,57],axis=0,inplace=True)
data

# In [ ]
# replace --- with nan
pd.set_option("future.no_silent_downcasting", True)
data = data.replace({'---': np.nan})
data

# In [ ]
# Replace NaNs with column median, skipping the first column
for col in data.columns[1:]:

  # Convert to float (ensure numeric type)
  data[col] = data[col].astype(float)

  # Compute median for the column
  median_value = data[col].median()

  # Replace NaN values with the column median
  data[col] = data[col].fillna(median_value)

data

# In [ ]
# extract to matrices
yearM = data.columns[1:].to_numpy().astype(float)
yearM

statesM = data.iloc[:,0]
statesM

M = data.iloc[:,1:].to_numpy()
np.round(M,2)

# In [ ]
# make some plots

fig,ax = plt.subplots(3,1,figsize=(8,5))

ax[0].plot(yearM,M.T)
ax[0].set_ylabel('M. rate (per 1k)')
ax[0].set_title('Marriage rates over time')

ax[1].plot(yearM,stats.zscore(M.T))
ax[1].set_ylabel('M. rate (per 1k)')
ax[1].set_title('M. rate (z-norm)')

# notice that x-axis is non-constant
ax[2].plot(yearM,np.mean(M,axis=0),'ks-',markerfacecolor='w',markersize=8)
ax[2].set_ylabel('M. rate (per 1k)')
ax[2].set_title('State-average')
ax[2].set_xlabel('Year')
# QUESTION: Is this the same as the US average?

plt.tight_layout()
plt.show()

plt.imshow(stats.zscore(M,axis=1),aspect='auto')
plt.xticks([])
plt.xlabel('Year')
plt.ylabel('State index')
plt.colorbar()
plt.show()

# In [ ]
# barplot of average marriage rate

# average over time
meanMarriageRate = np.mean(M,axis=1)

# sort index
sidx_M = np.argsort(meanMarriageRate)

fig = plt.figure(figsize=(12,4))
plt.bar(statesM.iloc[sidx_M],meanMarriageRate[sidx_M])
plt.xticks(rotation=90)
plt.ylabel('M. rate (per 1k)')
plt.title('Marriage rates per state')
plt.show()

# QUESTION:
#   Is Nevada a non-representative datapoint or an error?

# In [ ]
# show the correlation matrix

plt.imshow(np.corrcoef(M.T),vmin=.9,vmax=1,
             extent=[yearM[0],yearM[-1],yearM[-1],yearM[0]])
plt.colorbar()
plt.show()

# In [ ]
# PCA

pca = PCA().fit(M)

# scree plot
plt.plot(100*pca.explained_variance_ratio_,'ks-',markerfacecolor='w',markersize=10)
plt.ylabel('Percent variance explained')
plt.xlabel('Component number')
plt.title('PCA scree plot of marriage data')
plt.show()
100*pca.explained_variance_ratio_

# %% [markdown]
# # Repeat for divorce data

# In [ ]
# import the data
data = pd.read_excel(divorce_url,header=5)
data.drop([0,52,53,54,55,56,57],axis=0,inplace=True)
data = data.replace({'---': np.nan})

# Replace NaNs with column median
for col in data.columns[1:]:
  data[col] = data[col].astype(float)
  median_value = data[col].median()
  data[col] = data[col].fillna(median_value)

yearD = data.columns[1:].to_numpy().astype(float)
statesD = data.iloc[:,0]
D = data.iloc[:,1:].to_numpy()

# In [ ]
# make some plots
fig,ax = plt.subplots(3,1,figsize=(8,5))

ax[0].plot(yearD,D.T)
ax[0].set_ylabel('D. rate (per 1k)')
ax[0].set_title('Divorce rates over time')

ax[1].plot(yearD,stats.zscore(D.T))
ax[1].set_ylabel('D. rate (per 1k)')
ax[1].set_title('D. rate (z-norm)')

# notice that x-axis is non-constant
ax[2].plot(yearD,np.mean(D,axis=0),'ks-',markerfacecolor='w',markersize=8)
ax[2].set_ylabel('D. rate (per 1k)')
ax[2].set_title('State-average')
ax[2].set_xlabel('Year')
plt.tight_layout()
plt.show()

plt.imshow(stats.zscore(D,axis=1),aspect='auto')
plt.xticks([])
plt.xlabel('Year')
plt.ylabel('State index')
plt.show()





# barplot of average marriage rate
meanDivorceRate = np.mean(D,axis=1)
sidx_D = np.argsort(meanDivorceRate)

fig = plt.figure(figsize=(12,4))
plt.bar(statesD.iloc[sidx_D],meanDivorceRate[sidx_D])
plt.xticks(rotation=90)
plt.ylabel('D. rate (per 1k)')
plt.title('Divorce rates per state')
plt.show()






# show the correlation matrix
plt.imshow(np.corrcoef(D.T),vmin=.9,vmax=1,
             extent=[yearD[0],yearD[-1],yearD[-1],yearD[0]])
plt.colorbar()
plt.show()





# PCA
pca = PCA().fit(D)

# scree plot
plt.plot(pca.explained_variance_ratio_,'ks-',markerfacecolor='w',markersize=10)
plt.ylabel('Percent variance explained')
plt.xlabel('Component number')
plt.title('PCA scree plot of divorce data')
plt.show()

# In [ ]
# check if marriage and divorce datasets have the same year/state order

# should be zero
print( 'Comparison of year vectors: ')
print( np.sum(yearD-yearM) )

# should be TRUE
print('')
print( 'Comparison of states vectors: ')
print( statesM.equals(statesD) )
# ... uh oh...

# compare
tmpStateNames = pd.concat([statesM,statesD],axis=1)
print(tmpStateNames)

# find the difference
np.where(tmpStateNames.iloc[:,0] != tmpStateNames.iloc[:,1])

# In [ ]
# btw, you can also correlate over states

fig = plt.figure(figsize=(12,12))
plt.imshow(np.corrcoef(D),vmin=0,vmax=1)
plt.xticks(ticks=range(len(statesD)),labels=statesD,rotation=90)
plt.yticks(ticks=range(len(statesD)),labels=statesD)
plt.colorbar()
plt.show()

# %% [markdown]
# # Now for some inferential statistics

# In [ ]
# Correlate M and D over time per state


# Bonferroni corrected threshold
pvalThresh = .05/51


fig = plt.figure(figsize=(6,10))

color = 'rg'
for si in range(len(statesM)):

    # compute correlation
    r,p = stats.pearsonr(M[si,:],D[si,:])

    # plot the data point
    plt.plot([r,1],[si,si],'-',color=[.5,.5,.5])
    plt.plot(r,si,'ks',markerfacecolor=color[bool(p<pvalThresh)])

plt.ylabel('State')
plt.xlabel('Correlation')
plt.title('Marriage-divorce correlations per state')
plt.yticks(range(len(statesM)),labels=statesD)
plt.tick_params(axis='y',which='both',labelleft=False,labelright=True)
plt.xlim([-1,1])
plt.ylim([-1,51])
plt.plot([0,0],[-1,51],'k--')
plt.show()

# In [ ]
# have marriage/divorce rates really declined over time?

fig,ax = plt.subplots(2,1,figsize=(12,6))


# initialize slope differences vector
MvsD = np.zeros(len(statesM))

for rowi in range(len(statesM)):

    # run regression (includes the intercept!)
    bM,intercept,r,pM,seM = stats.linregress(yearM,M[rowi,:])
    bD,intercept,r,pD,seD = stats.linregress(yearM,D[rowi,:])

    # normalize beta coefficients
    bM = bM / seM
    bD = bD / seD

    # plot the slope values
    ax[0].plot([rowi,rowi],[bM,bD],'k')
    ax[0].plot(rowi,bM,'ko',markerfacecolor=color[bool(pM<pvalThresh)])
    ax[0].plot(rowi,bD,'ks',markerfacecolor=color[bool(pD<pvalThresh)])

    # plot the slope differences
    ax[1].plot([rowi,rowi],[bM-bD, 0],'k-',color=[.7,.7,.7])
    ax[1].plot([rowi,rowi],[bM-bD,bM-bD],'ko',color=[.7,.7,.7])

    # store the slope differences for subsequent t-test
    MvsD[rowi] = bM-bD



# make the plot look nicer
for i in range(2):
    ax[i].set_xticks(range(51))
    ax[i].set_xticklabels(statesD,rotation=90)
    ax[i].set_xlim([-1,51])
    ax[i].plot([-1,52],[0,0],'k--')

ax[0].set_ylabel('Decrease per year (norm.)')
ax[1].set_ylabel('$\Delta$M - $\Delta$D')



### ttest on whether the M-vs-D rates are really different
t,p = stats.ttest_1samp(MvsD,0)
df = len(MvsD)-1

# set the title
ax[1].set_title('Marriage vs. divorce: t(%g)=%g, p=%g'%(df,t,p))

plt.tight_layout()
plt.show()

# In [ ]

