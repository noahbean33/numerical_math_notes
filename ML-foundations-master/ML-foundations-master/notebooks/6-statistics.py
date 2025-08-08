# -*- coding: utf-8 -*-
# Auto-generated from '6-statistics.ipynb' on 2025-08-08T15:22:58
# Source: code cells extracted; markdown preserved as comments.

# %% [markdown]
# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/6-statistics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Intro to Statistics

# %% [markdown]
# This class, *Intro to Statistics*, builds on probability theory to enable us to quantify our confidence about how distributions of data are related to one another.
# 
# Through the measured exposition of theory paired with interactive examples, you’ll develop a working understanding of all of the essential statistical tests for assessing whether data are correlated with each other or sampled from different populations -- tests which frequently come in handy for critically evaluating the inputs and outputs of machine learning algorithms. You’ll also learn how to use regression to make predictions about the future based on training data.
# 
# The content covered in this class builds on the content of other classes in the *Machine Learning Foundations* series (linear algebra, calculus, and probability theory) and is itself foundational for the *Optimization* class.

# %% [markdown]
# Over the course of studying this topic, you'll:
# 
# * Develop an understanding of what’s going on beneath the hood of predictive statistical models and machine learning algorithms, including those used for deep learning.
# * Hypothesize about and critically evaluate the inputs and outputs of machine learning algorithms using essential statistical tools such as the t-test, ANOVA, and R-squared.
# * Use historical data to predict the future using regression models that take advantage of frequentist statistical theory (for smaller data sets) and modern machine learning theory (for larger data sets), including why we may want to consider applying deep learning to a given problem.

# %% [markdown]
# **Note that this Jupyter notebook is not intended to stand alone. It is the companion code to a lecture or to videos from Jon Krohn's [Machine Learning Foundations](https://github.com/jonkrohn/ML-foundations) series, which offer detail on the following:**
# 
# *Segment 1: Frequentist Statistics*
# 
# * Frequentist vs Bayesian Statistics
# * Review of Relevant Probability Theory
# * *z*-scores and Outliers
# * *p*-values
# * Comparing Means with t-tests
# * Confidence Intervals
# * ANOVA: Analysis of Variance
# * Pearson Correlation Coefficient
# * R-Squared Coefficient of Determination
# * Correlation vs Causation
# * Correcting for Multiple Comparisons
# 
# *Segment 2: Regression*
# 
# * Features: Independent vs Dependent Variables
# * Linear Regression to Predict Continuous Values
# * Fitting a Line to Points on a Cartesian Plane
# * Ordinary Least Squares
# * Logistic Regression to Predict Categories
# 
# *Segment 3: Bayesian Statistics*
# 
# * (Deep) ML vs Frequentist Statistics
# * When to use Bayesian Statistics
# * Prior Probabilities
# * Bayes’ Theorem
# * PyMC3 Notebook
# * Resources for Further Study of Probability and Statistics

# %% [markdown]
# ## Segment 1: Frequentist Statistics

# In [1]
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

# In [2]
np.random.seed(42)

# %% [markdown]
# ### Measures of Central Tendency

# %% [markdown]
# Measures of central tendency provide a summary statistic on the center of a given distribution, a.k.a., the "average" value of the distribution.

# In [3]
x = st.skewnorm.rvs(10, size=1000)

# In [4]
x[0:20]

# In [5]
fig, ax = plt.subplots()
_ = plt.hist(x, color = 'lightgray')

# %% [markdown]
# #### Mean

# %% [markdown]
# The most common measure of central tendency, synonomous with the term "average", is the **mean**, often symbolized with $\mu$ (population) or $\bar{x}$ (sample):

# %% [markdown]
# $$ \bar{x} = \frac{\sum_{i=1}^n x_i}{n} $$

# In [6]
xbar = x.mean()
xbar

# In [7]
fig, ax = plt.subplots()
plt.axvline(x = x.mean(), color='orange')
_ = plt.hist(x, color = 'lightgray')

# %% [markdown]
# #### Median

# %% [markdown]
# The second most common measure of central tendency is the **median**, the midpoint value in the distribution:

# In [8]
np.median(x)

# %% [markdown]
# The **mode** is least impacted by skew, but is typically only applicable to discrete distributions. For continuous distributions with skew (e.g., salary data), median is typically the choice measure of central tendency:

# In [9]
fig, ax = plt.subplots()
plt.axvline(x = np.mean(x), color='orange')
plt.axvline(x = np.median(x), color='green')
_ = plt.hist(x, color = 'lightgray')

# %% [markdown]
# ### Measures of Dispersion

# %% [markdown]
# #### Variance

# %% [markdown]
# $$ \sigma^2 = \frac{\sum_{i=1}^n (x_i-\bar{x})^2}{n} $$

# In [10]
x.var()

# %% [markdown]
# #### Standard Deviation

# %% [markdown]
# A straightforward derivative of variance is **standard deviation** (denoted with $\sigma$), which is convenient because its units are on the same scale as the values in the distribution:
# $$ \sigma = \sqrt{\sigma^2} $$

# In [11]
x.var()**(1/2)

# In [12]
sigma = x.std()
sigma

# In [13]
fig, ax = plt.subplots()
plt.axvline(x = xbar, color='orange')
plt.axvline(x = xbar+sigma, color='olivedrab')
plt.axvline(x = xbar-sigma, color='olivedrab')
_ = plt.hist(x, color = 'lightgray')

# %% [markdown]
# #### Standard Error

# %% [markdown]
# A further derivation of standard deviation is **standard error**, which is denoted with $\sigma_\bar{x}$:
# $$ \sigma_\bar{x} = \frac{\sigma}{\sqrt{n}} $$

# In [14]
sigma/(x.size)**(1/2)

# In [15]
st.sem(x) # defaults to 1 degree of freedom, which can be ignored with the larger data sets of ML

# In [16]
st.sem(x, ddof=0)

# %% [markdown]
# Standard error enables us to compare whether the means of two distributions differ *significantly*, a focus of *Intro to Stats*.

# %% [markdown]
# ### Gaussian Distribution

# %% [markdown]
# After Carl Friedrich Gauss. Also known as **normal distribution**:

# In [17]
x = np.random.normal(size=10000)

# In [18]
sns.set_style('ticks')

# In [19]
_ = sns.displot(x, kde=True)

# %% [markdown]
# When the normal distribution has a mean ($\mu$) of zero and standard deviation ($\sigma$) of one, as it does by default with the NumPy `normal()` method...

# In [20]
x.mean()

# In [21]
x.std()

# %% [markdown]
# ...it is a **standard normal distribution** (a.k.a., standard Gaussian distribution or ***z*-distribution**), which can be denoted as $\mathcal{N}(\mu, \sigma^2) = \mathcal{N}(0, 1)$ (noting that $\sigma^2 = \sigma$ here because $1^2 = 1$).

# %% [markdown]
# Normal distributions are by far the most common distribution in statistics and machine learning. They are typically the default option, particularly if you have limited information about the random process you're modeling, because:
# 
# 1. Normal distributions assume the greatest possible uncertainty about the random variable they represent (relative to any other distribution of equivalent variance). Details of this are beyond the scope of this tutorial.
# 2. Simple and very complex random processes alike are, under all common conditions, normally distributed when we sample values from the process. Since we sample data for statistical and machine learning models alike, this so-called **central limit theorem** (covered next) is a critically important concept.

# %% [markdown]
# ### The Central Limit Theorem

# %% [markdown]
# To develop a functional understanding of the CLT, let's sample some values from our normal distribution:

# In [22]
x_sample = np.random.choice(x, size=10, replace=False)
x_sample

# %% [markdown]
# The mean of a sample isn't always going to be close to zero with such a small sample:

# In [23]
x_sample.mean()

# %% [markdown]
# Let's define a function for generating **sampling distributions** of the mean of a given input distribution:

# In [24]
def sample_mean_calculator(input_dist, sample_size, n_samples):
    sample_means = []
    for i in range(n_samples):
        sample = np.random.choice(input_dist, size=sample_size, replace=False)
        sample_means.append(sample.mean())
    return sample_means

# In [25]
sns.displot(sample_mean_calculator(x, 10, 10), color='green', kde=True)
_ = plt.xlim(-1.5, 1.5)

# %% [markdown]
# The more samples we take, the more likely that the sampling distribution of the means will be normally distributed:

# In [26]
sns.displot(sample_mean_calculator(x, 10, 1000), color='green', kde=True)
_ = plt.xlim(-1.5, 1.5)

# %% [markdown]
# The larger the sample, the tighter the sample means will tend to be around the population mean:

# In [27]
sns.displot(sample_mean_calculator(x, 100, 1000), color='green', kde=True)
_ = plt.xlim(-1.5, 1.5)

# In [28]
sns.displot(sample_mean_calculator(x, 1000, 1000), color='green', kde=True)
_ = plt.xlim(-1.5, 1.5)

# %% [markdown]
# #### Sampling from a skewed distribution

# In [29]
s = st.skewnorm.rvs(10, size=10000)

# In [30]
_ = sns.displot(s, kde=True)

# In [31]
_ = sns.displot(sample_mean_calculator(s, 10, 1000), color='green', kde=True)

# In [32]
_ = sns.displot(sample_mean_calculator(s, 1000, 1000), color='green', kde=True)

# %% [markdown]
# #### Sampling from a multimodal distribution

# In [33]
m = np.concatenate((np.random.normal(size=5000), np.random.normal(loc = 4.0, size=5000)))

# In [34]
_ = sns.displot(m, kde=True)

# In [35]
_ = sns.displot(sample_mean_calculator(m, 1000, 1000), color='green', kde=True)

# %% [markdown]
# #### Sampling from uniform

# %% [markdown]
# Even sampling from the highly non-normal uniform distribution, the sampling distribution comes out normal:

# In [36]
u = np.random.uniform(size=10000)

# In [37]
_ = sns.displot(u)

# In [38]
_ = sns.displot(sample_mean_calculator(u, 1000, 1000), color='green', kde=True)

# %% [markdown]
# Therefore, with large enough sample sizes, we can assume the sampling distribution of the means will be normally distributed, allowing us to apply statistical and ML models that are configured for normally distributed noise, which is often the default assumption.
# 
# As an example, the "*t*-test" (covered shortly in *Intro to Stats*) allows us to infer whether two samples come from different populations (say, an experimental group that receives a treatment and a control group that receives a placebo). Thanks to the CLT, we can use this test even if we have no idea what the underlying distributions of the populations being tested are, which may be the case more frequently than not.

# %% [markdown]
# ### z-scores

# %% [markdown]
# Assuming normally-distributed data, a z-score indicates how many standard deviations away from the mean a data point (say, $x_i$) is:
# $$ z = \frac{x_i-\mu}{\sigma} $$

# %% [markdown]
# That is, the formula *standardizes* a given score $x_i$ to the (standard normal) *z*-distribution. (As we covered in *Probability & Information Theory*, you could standardize any normal distribution to a mean of zero and standard deviation of one by subtracting its original mean and then dividing by its original standard deviation.)

# %% [markdown]
# For example, let's say you get 85% on a CS101 exam. Sounds like a pretty good score and you did extremely well relative to your peers if the mean was 60% with a standard deviation of 10%:

# In [39]
x_i = 85
mu = 60
sigma = 10

# In [40]
x = np.random.normal(mu, sigma, 10000)

# In [41]
sns.displot(x, color='gray')
ax.set_xlim(0, 100)
plt.axvline(mu, color='orange')
for v in [-3, -2, -1, 1, 2, 3]:
    plt.axvline(mu+v*sigma, color='olivedrab')
_ = plt.axvline(x_i, color='purple')

# %% [markdown]
# Your z-score is 2.5 standard deviations above the mean:

# In [42]
z = (x_i - mu)/sigma
z

# %% [markdown]
# Or using our simulated class of 10k CS101 students:

# In [43]
z = (x_i - np.mean(x))/np.std(x)
z

# %% [markdown]
# Less than one percent of the class outperformed you:

# In [44]
len(np.where(x > 85)[0])

# In [45]
100*69/10000

# In [46]
np.percentile(x, 99)

# %% [markdown]
# In contrast, if the mean score of your peers is 90 and the standard deviation is 2:

# In [47]
mu = 90
sigma = 2

# In [48]
y = np.random.normal(mu, sigma, 10000)

# In [49]
sns.displot(y, color='gray')
plt.axvline(mu, color='orange')
for v in [-3, -2, -1, 1, 2, 3]:
    plt.axvline(mu+v*sigma, color='olivedrab')
_ = plt.axvline(x_i, color='purple')

# %% [markdown]
# Your z-score is 2.5 standard deviations *below* the mean (!):

# In [50]
z = (x_i - mu)/sigma
z

# %% [markdown]
# Or using our simulated class of 10k CS101 students:

# In [51]
z = (x_i - np.mean(y))/np.std(y)
z

# %% [markdown]
# In which case, over 99% of the class outperformed you:

# In [52]
len(np.where(y > 85)[0])

# In [53]
100*9933/10000

# %% [markdown]
# A mere 67 folks attained worse:

# In [54]
10000-9933

# In [55]
np.percentile(y, 1)

# %% [markdown]
# A frequentist convention is to consider a data point that lies further than three standard deviations from the mean to be an **outlier**.
# 
# It's a good idea to individually investigate outliers in your data as they may represent an erroneous data point (e.g., some data by accident, a data-entry error, or a failed experiment) that perhaps should be removed from further analysis (especially, as outliers can have an outsized impact on statistics including mean and correlation). It may even tip you off to a major issue with your data-collection methodology or your ML model that can be resolved or that you could have a unit test for.

# %% [markdown]
# **Exercises**
# 
# 1. You clean and jerk 100kg in a weightlifting competition. The mean C&J weight at the competition is 100kg. What's your z-score for the C&J?
# 2. You snatch 100kg in the same competition. The mean snatch weight is 80kg with a standard deviation of 10kg. What's your z-score for the snatch?
# 3. In olympic weightlifting, your overall score is the sum total of your C&J and snatch weights. The mean of these totals across competitors is 180kg with a standard deviation of 5kg. What's your overall z-score in the competition?
# 
# **Spoiler alert**: Solutions below

# In [55]

# In [55]

# In [55]

# %% [markdown]
# **Solutions**
# 1. zero
# 2. two
# 3. four (you may have won the meet!)

# %% [markdown]
# ### *p*-values

# %% [markdown]
# These quantify the *p*robability that a given observation would occur by chance alone.
# 
# For example, we saw above that with our simulated 10k exam results, only 69 folks attained a *z*-score above 2.5 and only 67 (=10000-9993) attained a *z*-score below -2.5. Thus, if we were to randomly sample one of the 10k CS101 exam results, we would expect it to be outside of 2.5 (i.e., +/- 2.5) standard deviations only 1.36% of the time:
# $$ \frac{69+67}{10000} = 0.0136 = 1.36\% $$

# %% [markdown]
# Equivalent to increasing our CS101 class size from 10k toward infinity, the probability of a score being further than 2.5 standard deviations from the mean of a normal distribution can be determined with the distribution's *cumulative distribution function* (CDF):

# In [56]
p_below = st.norm.cdf(-2.5)
p_below

# In [57]
p_below*10000

# In [58]
sns.displot(y, color='gray')
_ = plt.axvline(mu-2.5*sigma, color='blue')

# In [59]
st.norm.cdf(2.5)

# In [60]
p_above = 1-st.norm.cdf(2.5)
p_above

# In [61]
p_above*10000

# In [62]
sns.displot(y, color='gray')
_ = plt.axvline(mu+2.5*sigma, color='blue')

# In [63]
p_outside = p_below + p_above
p_outside

# In [64]
p_outside*10000

# In [65]
sns.displot(y, color='gray')
plt.axvline(mu+2.5*sigma, color='blue')
_ = plt.axvline(mu-2.5*sigma, color='blue')

# %% [markdown]
# In other words, assuming a normal distribution, the probability (the *p*-value) of a sampled value being at least 2.5 standard deviations away from the mean by chance alone is $p \approx .0124$.

# %% [markdown]
# The frequentist convention is that if a *p*-value is less than .05, we can say that it is a "statistically significant" observation. We typically denote this significance threshold with $\alpha$, e.g., $\alpha = .05$.
# 
# For example, with a fair coin, the probability of throwing six heads *or* six tails in a six-coin-flip experiment is 0.03125 ($p = 0.015625$ for *either of* six heads or six tails). Refer back to the `coinflip_prob()` method from the [*Probability* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/5-probability.ipynb) for proof.
# 
# If a friend of yours hands you a coin, the **null hypothesis** (the baseline assumed by the fair-toss distribution) would be that the coin is fair. If you test this coin by flipping it six times and it comes up heads on all six or tails on all six, this observation would suggest that you should *reject the null hypothesis* because chance alone would facilitate such an observation less than 5% of the time, i.e., $p < .05$.

# %% [markdown]
# The *z*-scores corresponding to $\alpha = .05$ can be obtained from the normal distribution's *percent point function* (PPF), which facilitates the inverse of the CDF. To capture 95% of the values around the mean, we leave 2.5% at the bottom of the distribution and 2.5% at the top:

# In [66]
st.norm.ppf(.025)

# In [67]
st.norm.ppf(.975)

# %% [markdown]
# Thus, at the traditional $\alpha = .05$, a sampled value with *z*-score less than -1.96 or greater than 1.96 would be considered statistically significant.

# In [68]
sns.displot(y, color='gray')
plt.axvline(mu+1.96*sigma, color='darkred')
_ = plt.axvline(mu-1.96*sigma, color='darkred')

# %% [markdown]
# With a stricter threshold, say $\alpha = .01$:

# In [69]
st.norm.ppf(.005)

# In [70]
st.norm.ppf(.995)

# In [71]
sns.displot(y, color='gray')

plt.axvline(mu+1.96*sigma, color='darkred')
plt.axvline(mu-1.96*sigma, color='darkred')

plt.axvline(mu+2.56*sigma, color='black')
_ = plt.axvline(mu-2.56*sigma, color='black')

# %% [markdown]
# (Time-permitting, a discussion of two-tailed vs one-tailed *p*-value tests would be informative here.)

# %% [markdown]
# **Exercises**
# 
# 1. What are the *p*-values associated with your weightlifting results from the three preceding exercises?
# 2. With the standard $\alpha = .05$, which of the three weightlifting results are "statistically significant"?
# 
# **Spoiler alert**: Solutions below

# In [71]

# In [71]

# In [71]

# %% [markdown]
# **Solutions**

# %% [markdown]
# 1a. This result is at the mean, which is also the median for a normal distribution; exactly half of the values are above as they are below. This corresponds to the highest possible $p$-value, $p=1$, because any value in the distribution is guaranteed to be above it or below it:

# In [72]
p_below = st.norm.cdf(0)
p_below

# In [73]
p_above = 1-st.norm.cdf(0)
p_above

# In [74]
p_below + p_above

# %% [markdown]
# More generally:

# In [75]
def p_from_z(my_z):
    return 2 * st.norm.cdf(-abs(my_z))

# In [76]
p_from_z(0)

# %% [markdown]
# 1b. The probability of a value being below $z = -2$ is:

# In [77]
p_below = st.norm.cdf(-2)
p_below

# %% [markdown]
# ...and the probability of a value being above $z=2$ is the same:

# In [78]
p_above = 1-st.norm.cdf(2)
p_above

# %% [markdown]
# Therefore, the *p*-value -- the probability that a value is below $z=-2$ or above $z=2$ -- is:

# In [79]
p_below + p_above

# In [80]
p_from_z(2)

# %% [markdown]
# 1c. Following the same calculations as we did for 1b, the *p*-value for an observation 4 standard deviations away from the mean is:

# In [81]
p_from_z(4)

# %% [markdown]
# ...which is about 0.0000633:

# In [82]
0.0000633

# %% [markdown]
# (Incidentally, very small *p* values are often reported as **negative log *P*** values as these are much easier to read...)

# In [83]
-np.log10(6.33e-05)

# %% [markdown]
# 2. The absolute value of the *z*-score for your snatch as well as your combined score is greater than 1.96 so they're both "statistically significant". Your performance on the clean and jerk could not have been less significant!

# %% [markdown]
# ### Comparing Means with *t*-tests

# %% [markdown]
# Where *z*-scores apply to *individual values* only, *t*-tests enables us to compare (the mean of) a sample of *multiple values* to a reference mean.

# %% [markdown]
# #### Student's Single-Sample *t*-test

# %% [markdown]
# Named after William Sealy Gosset, an Oxford-trained scientist and mathematician, who became a stout yield statistician for Guinness in Dublin (from 1899 to his fatal heart attack in 1937 shortly after being promoted to head brewer). Alongside sabbaticals in Karl Pearson's UCL Biometric Laboratory, Gosset published under the pseudonym Student (including on the *t*-test, starting in 1908) as it was against Guinness policy to publish.

# %% [markdown]
# Recalling the formula for calculating a *z*-score:
# $$ z = \frac{x_i-\mu}{\sigma} $$

# %% [markdown]
# The **single-sample *t*-test** is a variation on the theme and is defined by:
# $$ t = \frac{\bar{x} - \mu_0}{s_{\bar{x}}} $$
# Where:
# * $\bar{x}$ is the sample mean
# * $\mu_0$ is a reference mean, e.g., known population mean or "null hypothesis" mean
# * $s_{\bar{x}}$ is the sample standard error

# %% [markdown]
# Let's say you're the head brewer at Guinness. Your baseline brewing process yields 50L of stout. Using a new genetically-modified yeast, you obtain the following yields (all in liters) in four separate experiments:

# In [84]
x = [48, 50, 54, 60]

# %% [markdown]
# We can obtain the *t*-statistic for this sample as follows:

# In [85]
xbar = np.mean(x)
xbar

# In [86]
sx = st.sem(x)
sx

# In [87]
t = (xbar-50)/sx
t

# %% [markdown]
# We can convert the *t*-value into a *p*-value using Student's *t*-distribution (similar to the normal *z*-distribution, but varies based on number of data points in sample; see [here](https://en.wikipedia.org/wiki/Student%27s_t-distribution) for more detail):

# In [88]
def p_from_t(my_t, my_n):
    return 2 * st.t.cdf(-abs(my_t), my_n-1) # 2nd arg to t.cdf() is "degrees of freedom"

# In [89]
p_from_t(t, len(x))

# %% [markdown]
# (An illustration of **degrees of freedom**: If we know the mean of the array `x`, three of its four values can vary freely. That is, if we know three of the values in the array, the fourth has no "freedom"; it must be a specific value. Thus, the most common situation with statistical tests is that we have *n*-1 degrees of freedom.)

# %% [markdown]
# For everyday usage, however, we can rely on the SciPy `ttest_1samp()` method:

# In [90]
st.ttest_1samp(x, 50)

# %% [markdown]
# #### Welch's Independent *t*-test

# %% [markdown]
# In ordinary circumstances, if we have two samples whose means we'd like to compare, we use an **independent *t*-test**.

# In [91]
penguins = sns.load_dataset('penguins').dropna() # some rows are missing data

# In [92]
penguins

# In [93]
np.unique(penguins.species, return_counts=True)

# In [94]
adelie = penguins[penguins.species == 'Adelie']

# In [95]
adelie

# In [96]
np.unique(adelie.island, return_counts=True)

# In [97]
np.unique(adelie.sex, return_counts=True)

# In [98]
_ = sns.boxplot(x='island', y='body_mass_g', hue='sex', data=adelie)

# %% [markdown]
# Mass doesn't appear to vary by island, so we can feel comfortable grouping the data together by island. Weight does, however, appear to vary by sex so let's take a closer look:

# In [99]
f = adelie[adelie.sex == 'Female']['body_mass_g'].to_numpy()/1000
f

# In [100]
m = adelie[adelie.sex == 'Male']['body_mass_g'].to_numpy()/1000
m

# In [101]
fbar = f.mean()
fbar

# In [102]
mbar = m.mean()
mbar

# %% [markdown]
# To quantify whether males weigh significantly more than females, we can use the **Welch *t*-test**, devised by the 20th c. British statistician Bernard Lewis Welch:
# $$ t = \frac{\bar{x} - \bar{y}}{\sqrt{\frac{s^2_x}{n_x} + \frac{s^2_y}{n_y}}} $$
# Where:
# * $\bar{x}$ and $\bar{y}$ are the sample means
# * $s^2_x$ and $s^2_y$ are the sample variances
# * $n_x$ and $n_y$ are the sample sizes

# %% [markdown]
# **N.B.**: Student's independent *t*-test is markedly more popular than Welch's, but Student's assumes equal population variances (i.e., $\sigma^2_x \approx \sigma^2_y$), making it less robust. In case you're curious, Student's formula is the same as Welch's, except that it uses a pooled variance $s^2_p$ in place of individual sample variances ($s^2_x$ and $s^2_y$). You can read more about it [here](https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test).

# In [103]
sf = f.var(ddof=1)
sm = m.var(ddof=1)

# In [104]
nf = f.size
nm = m.size

# In [105]
t = (fbar-mbar)/(sf/nf + sm/nm)**(1/2)
t

# %% [markdown]
# Degrees of freedom for calculating the *p*-value are estimated using the [Welch–Satterthwaite equation](https://en.wikipedia.org/wiki/Welch–Satterthwaite_equation), which we won't detail but is defined as:

# In [106]
def ws_eqn(sx, sy, nx, ny):
    return (sx / nx + sy / ny)**2 / (sx**2 / (nx**2 * (nx - 1)) + sy**2 / (ny**2 * (ny - 1)))

# In [107]
df = ws_eqn(sf, sm, nf, nm)
df

# In [108]
p = 2 * st.t.cdf(-abs(t), df) # or p_from_t(t, df+1)
p

# In [109]
p_from_t(t, df+1)

# In [110]
-np.log10(p)

# In [111]
st.ttest_ind(f, m, equal_var=False)

# %% [markdown]
# #### Student's Paired *t*-test

# %% [markdown]
# Occasionally, we have two vectors where each element in vector *x* has a corresponding element in vector *y*.
# 
# For example, we could run an experiment where Alzheimer's disease patients receive a drug on one day (experimental condition) and a sugar pill placebo (control condition) on another day. We can then measure the patients' forgetfulness on both days to test whether the drug has a significant impact on memory.
# 
# For a given sample size, such a paired *t*-test is more powerful relative to an unpaired (independent) *t*-test because the variance of *x* is directly related to the variance in *y*: A severe Alzheimer's patient will tend to be relatively forgetful on both days, while a mild Alzheimer's patient will tend to be relatively unforgetful on both days. With paired samples, we capture this power by comparing the *difference* between *x* and *y*, e.g., the difference in forgetfulness for a given patient when given the drug relative to when given the sugar pill.
# 
# In contrast, consider the penguin dataset, wherein we wouldn't be able to obviously pair a given male penguin with a correponding female penguin. Or consider a situation where we provide a drug to one set of Alzheimer's patients while we provide a placebo to an entire different (an independent) group of patients. Indeed, with an independent *t*-test we could even have different sample sizes in the two groups whereas this is impossible with a paired *t*-test.

# %% [markdown]
# Here's an example:

# In [112]
exercise = sns.load_dataset('exercise')
exercise

# %% [markdown]
# There are 30 people in the dataset, with their pulse taken at three different time points in an experiment (i.e, after one, 15, and 30 minutes). Ten people were assigned to each of three activity groups:

# In [113]
np.unique(exercise.kind, return_counts=True)

# %% [markdown]
# Within each of those activity groups, half of the participants are on a low-fat diet while the other half are on a no-fat diet:

# In [114]
np.unique(exercise.diet, return_counts=True)

# %% [markdown]
# For simplicity, let's only consider one of the six experimental groups, say the walking, no-fat dieters:

# In [115]
walk_no = exercise[(exercise.diet == 'no fat') & (exercise.kind == 'walking')]
walk_no

# %% [markdown]
# (Note how participant 16 has a relatively low heart rate at all three timepoints, whereas participant 20 has a relatively high heart rate at all three timepoints.)

# In [116]
_ = sns.boxplot(x='time', y='pulse', data=walk_no)

# In [117]
min1 = walk_no[walk_no.time == '1 min']['pulse'].to_numpy()
min1

# In [118]
min1.mean()

# In [119]
min15 = walk_no[walk_no.time == '15 min']['pulse'].to_numpy()
min15

# In [120]
min15.mean()

# In [121]
min30 = walk_no[walk_no.time == '30 min']['pulse'].to_numpy()
min30

# In [122]
min30.mean()

# %% [markdown]
# (With paired samples, we can plot the values in a scatterplot, which wouldn't make any sense for independent samples, e.g.:)

# In [123]
sns.scatterplot(x=min1, y=min15)
plt.title('Heart rate of no-fat dieters (beats per minute)')
plt.xlabel('After 1 minute walking')
_ = plt.ylabel('After 15 minutes walking')

# %% [markdown]
# To assess whether the mean heart rate varies significantly after one minute of walking relative to after 15 minutes, we can use Student's **paired-sample** (a.k.a., **dependent**) *t*-test:
# $$ t = \frac{\bar{d} - \mu_0}{s_\bar{d}} $$
# Where:
# * $d$ is a vector of the differences between paired samples $x$ and $y$
# * $\bar{d}$ is the mean of the differences
# * $\mu_0$ will typically be zero, meaning the null hypothesis is that there is no difference between $x$ and $y$
# * $s_\bar{d}$ is the standard error of the differences

# %% [markdown]
# (Note how similar to single-sample *t*-test formula.)

# In [124]
d = min15 - min1
d

# In [125]
dbar = d.mean()
dbar

# In [126]
sd = st.sem(d)
sd

# In [127]
t = (dbar-0)/sd
t

# In [128]
p_from_t(t, d.size)

# In [129]
st.ttest_rel(min15, min1)

# %% [markdown]
# In contrast, if we were to put the same values into an independent *t*-test...

# In [130]
st.ttest_ind(min15, min1, equal_var=False)

# %% [markdown]
# #### Machine Learning Examples

# %% [markdown]
# * Single-sample: Does my stochastic model tend to be more accurate than an established benchmark?
# * Independent samples: Does my model have unwanted bias in it, e.g., do white men score higher than other demographic groups with HR model?
# * Paired samples: Is new TensorFlow.js model significantly faster? (paired by browser / device)

# %% [markdown]
# **Exercises**
# 1. You run four additional experiments with your GMO brewing yeast and now have the following eight stout yields: `[48, 50, 54, 60, 49, 55, 59, 62]`. What is the *t*-statistic and is it significantly different from the 50L-yielding baseline process?
# 2. Does the flipper length of Adélie penguins from Dream island vary significantly by sex?
# 2. Was the heart rate of low-fat dieters different after one minute of rest relative to after 15 minutes of rest?
# 
# **Spoiler alert**: Solutions below

# In [130]

# In [130]

# In [130]

# %% [markdown]
# **Solutions**
# 1. The GMO yeast yields a mean of 54.6L, which is significantly more stout than the baseline process, *t*(7) = 2.45, $p < .05$.

# In [131]
st.ttest_1samp([48, 50, 54, 60, 49, 55, 59, 62], 50)

# %% [markdown]
# 2. On Dream island, the flippers of male Adélie penguins (191.9 mm) are significantly longer than those of females (187.9 mm), *t* = 2.4, *p* < .05.

# In [132]
_ = sns.boxplot(x='island', y='flipper_length_mm', hue='sex', data=adelie)

# In [133]
f = adelie[(adelie.sex == 'Female') & (adelie.island == 'Dream')]['flipper_length_mm'].to_numpy()
m = adelie[(adelie.sex == 'Male') & (adelie.island == 'Dream')]['flipper_length_mm'].to_numpy()

# In [134]
f.mean()

# In [135]
m.mean()

# In [136]
tp = st.ttest_ind(f, m, equal_var=False)
tp

# In [137]
tp.pvalue

# %% [markdown]
# 3. The heart rate of low-fat dieters did not change significantly after one minute of rest (88.6 bpm) relative to after 15 minutes of rest (89.6 bpm), *t*=2.2, *p* = .09.

# In [138]
rest_lo = exercise[(exercise.diet == 'low fat') & (exercise.kind == 'rest')]

# In [139]
_ = sns.boxplot(x='time', y='pulse', data=rest_lo)

# In [140]
min1 = rest_lo[rest_lo.time == '1 min']['pulse'].to_numpy()
min1.mean()

# In [141]
min15 = rest_lo[rest_lo.time == '15 min']['pulse'].to_numpy()
min15.mean()

# In [142]
st.ttest_rel(min15, min1)

# %% [markdown]
# ### Confidence Intervals

# %% [markdown]
# When examining sample means as we have been for the *t*-test, a useful statistical tool is the **confidence interval** (CI), which we for example often see associated with polling results when there's an upcoming election. CIs allow us to make statements such as "there is a 95% chance that the population mean lies within this particular range of values".

# %% [markdown]
# We can calculate a CI by rearranging the *z*-score formula:
# $$ \text{C.I.} = \bar{x} \pm z \frac{s}{\sqrt{n}} $$
# Where:
# * $\bar{x}$ is the sample mean
# * $s$ is the sample standard deviation
# * $n$ is the sample size
# * $z$ corresponds to a *z*-score threshold (e.g., the most common 95% CI is $z \pm 1.960$; other popular ones are the 90% CI at $z \pm 1.645$ and the 99% CI at $z \pm 2.576$)

# %% [markdown]
# For example, to find the 95% confidence interval for the true mean yield of our GMO yeast:

# In [143]
x = np.array([48, 50, 54, 60, 49, 55, 59, 62])

# In [144]
xbar = x.mean()
s = x.std()
n = x.size

# In [145]
z = 1.96

# In [146]
def CIerr_calc(my_z, my_s, my_n):
    return my_z*(my_s/my_n**(1/2))

# In [147]
CIerr = CIerr_calc(z, s, n)

# In [148]
CIerr

# In [149]
xbar + CIerr

# In [150]
xbar - CIerr

# %% [markdown]
# Therefore, there's a 95% chance that the true mean yield of our GMO yeast lies in the range of 51.2 to 58.1 liters. Since this CI doesn't overlap with the established baseline mean of 50L, this corresponds to stating that the GMO yield is significantly greater than the baseline where $\alpha = .05$, as we already determined:

# In [151]
fig, ax = plt.subplots()
plt.ylabel('Stout Yield (L)')
plt.grid(axis='y')
ax.errorbar(['GMO'], [xbar], [CIerr], fmt='o', color='green')
_ = ax.axhline(50, color='orange')

# %% [markdown]
# Similarly, we can compare several sample means with CIs. Using our penguins again:

# In [152]
fCIerr = CIerr_calc(z, sf, nf)
mCIerr = CIerr_calc(z, sm, nm)

# In [153]
fig, ax = plt.subplots()
plt.title('Adélie Penguins')
plt.ylabel('Weight (kg)')
plt.grid(axis='y')
_ = ax.errorbar(['female', 'male'], [fbar, mbar], [fCIerr, mCIerr],
                fmt='.', color='green')

# %% [markdown]
# The CIs are nowhere near overlapping, corresponding to the extremely significant (-log*P* $\approx 25$) difference in penguin weight.

# %% [markdown]
# In contrast, the CIs for female penguins from the three islands...

# In [154]
t = adelie[(adelie.sex == 'Female') & (adelie.island == 'Torgersen')]['body_mass_g'].to_numpy()/1000
b = adelie[(adelie.sex == 'Female') & (adelie.island == 'Biscoe')]['body_mass_g'].to_numpy()/1000
d = adelie[(adelie.sex == 'Female') & (adelie.island == 'Dream')]['body_mass_g'].to_numpy()/1000

# In [155]
means = [t.mean(), b.mean(), d.mean()]

# In [156]
s_t, sb, sd = t.var(ddof=1), b.var(ddof=1), d.var(ddof=1) # s_t to disambiguate stats package

# In [157]
nt, nb, nd = t.size, b.size, d.size

# In [158]
CIerrs = [CIerr_calc(z, s_t, nt), CIerr_calc(z, sb, nb), CIerr_calc(z, sd, nd)]

# In [159]
fig, ax = plt.subplots()
plt.title('Female Adélie Penguins')
plt.ylabel('Weight (kg)')
plt.grid(axis='y')
_ = ax.errorbar(['Torgersen', 'Biscoe', 'Dream'], means, CIerrs,
                fmt='o', color='green')

# %% [markdown]
# ### ANOVA: Analysis of Variance

# %% [markdown]
# **Analysis of variance** (ANOVA) enables us to compare more than two samples (e.g., all three islands in the case of penguin weight) in a single statistical test.

# %% [markdown]
# To apply ANOVA, we must make three assumptions:
# 1. Independent samples
# 2. Normally-distributed populations
# 3. *Homoscedasticity*: Population standard deviations are equal

# %% [markdown]
# While not especially complicated under the hood (you can dig into the formulae [here](https://en.wikipedia.org/wiki/Analysis_of_variance#Logic)), ANOVA might be the least widely-applicable topic within *Intro to Stats* to ML so in the interest of time, we'll skip straight to the Python code:

# In [160]
st.f_oneway(t, b, d)

# %% [markdown]
# ### Pearson Correlation Coefficient

# %% [markdown]
# If we have two vectors of the same length, $x$ and $y$, where each element of $x$ is paired with the corresponding element of $y$, **covariance** provides a measure of how related the variables are to each other:
# $$ \text{cov}(x, y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) }{n} $$

# %% [markdown]
# A drawback of covariance is that it confounds the relative scale of two variables with a measure of the variables' relatedness. **Correlation** builds on covariance and overcomes this drawback via rescaling, thereby measuring (linear) relatedness exclusively. Correlation is much more common because of this difference.
# 
# The correlation coefficient (developed by Karl Pearson in the 20th c. though known in the 19th c.) is often denoted with $r$ or $\rho$ and is defined by:
# $$ \rho_{x,y} = \frac{\text{cov}(x,y)}{\sigma_x \sigma_y} $$

# In [161]
iris = sns.load_dataset('iris')
iris

# In [162]
x = iris.sepal_length
y = iris.petal_length

# In [163]
sns.set_style('darkgrid')

# In [164]
_ = sns.scatterplot(x=x, y=y)

# In [165]
n = iris.sepal_width.size

# In [166]
xbar, ybar = x.mean(), y.mean()

# In [167]
product = []
for i in range(n):
    product.append((x[i]-xbar)*(y[i]-ybar))

# In [168]
cov = sum(product)/n
cov

# In [169]
r = cov/(np.std(x)*np.std(y))
r

# %% [markdown]
# We reached this point in *Probability*. Now, as for how to determine a *p*-value, we first calculate the *t*-statistic...
# $$ t = r \sqrt{\frac{n-2}{1-r^2}} $$
# 
# (This formula standardizes the correlation coefficient, taking into account the sample size *n* and the strength of the relationship *r*, to produce a *t*-statistic that follows [Student's *t*-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution).)

# In [170]
t = r*((n-2)/(1-r**2))**(1/2)
t

# %% [markdown]
# ...which we can convert to a *p*-value as we've done several times above:

# In [171]
p = p_from_t(t, n-1)
p

# In [172]
-np.log10(p)

# %% [markdown]
# This confirms that iris sepal length's positive correlation with petal length is (extremely!) statistically significant.

# %% [markdown]
# All of the above can be done in a single line with SciPy's `pearsonr()` method:

# In [173]
st.pearsonr(x, y)

# %% [markdown]
# And, for reference, here's a correlation that is not significant ($r \approx 0$):

# In [174]
_ = sns.scatterplot(x=iris.sepal_length, y=iris.sepal_width)

# In [175]
st.pearsonr(iris.sepal_length, iris.sepal_width)

# %% [markdown]
# ### The Coefficient of Determination

# %% [markdown]
# ...also known as $r^2$, this is the proportion of variance in one variable explained by another.
# 
# It can range from 0 to 1 and it is simply the square of the Pearson $r$:

# In [176]
rsq = r**2
rsq

# %% [markdown]
# In this case, it indicates that 76% of the variance in iris petal length can be explained by sepal length. (This is easier to understand where one variable could straightforwardly drive variation in the other; more on that in Segment 2.)

# %% [markdown]
# For comparison, only 1.4% of the variance in sepal width can be explained by sepal length:

# In [177]
st.pearsonr(iris.sepal_length, iris.sepal_width)[0]**2

# %% [markdown]
# ### Correlation vs Causation

# %% [markdown]
# Correlation doesn't imply **causation** on its own. E.g., sepal length and petal length are extremely highly correlated, but this doesn't imply that sepal length causes petal length or vice versa. (Thousands of spurious correlations are provided [here](https://www.tylervigen.com/spurious-correlations) for your amusement.)

# %% [markdown]
# There is a lot to causality and I recommend Judea Pearl's [*Causality*](http://bayes.cs.ucla.edu/BOOK-2K/), the classic technical text on the topic, if you're keen to explore this in depth. [*The Book of Why*](http://bayes.cs.ucla.edu/WHY/) is an exceptional lay alternative by the same author.

# %% [markdown]
# In brief, three criteria are required for inferring causal relationships:
# 
# 1. **Covariation**: Two variables vary together (this criterion is satisfied by sepal and petal length)
# 2. **Temporal precedence**: The affected variable must vary *after* the causal variable is varied.
# 3. **Elimination of extraneous variables**: We must be sure no third variable is causing the variation. This can be tricky for data we obtained through observation alone, but easier when we can control the causal variable, e.g., with (ideally double-blind) randomized control trials.

# %% [markdown]
# Some examples of where we could infer causality from correlation in ML:
# * Additional neurons --> higher accuracy
# * Additional servers or RAM --> shorter inference time
# * Removal of pronouns --> less demographic bias in model

# %% [markdown]
# ### Correcting for Multiple Comparisons

# %% [markdown]
# A major issue with frequentist statistics is the issue of multiple comparisons:
# 
# * If you perform 20 statistical tests where there is no real effect (i.e., the null hypothesis is true), then we would expect one of them to come up significant by chance alone (i.e., a *false positive* or *Type I error*).
# * If you perform a hundred tests in such a circumstance, then you should expect five false positives.

# %% [markdown]
# The most straightforward, and indeed the most widely-used, solution is the **Bonferroni correction** (named after the 20th c. Italian mathematician Carlo Emilio Bonferroni). Assuming, we'd like an overall $\alpha = .05$:
# 
# * If we're planning on conducting ten tests ($m=10$), the significance threshold for each individual test is $\frac{\alpha}{m} = \frac{.05}{10} = .005$
# * With 20 tests, it's $\frac{\alpha}{m} = \frac{.05}{20} = .0025$
# * With 100 tests, it's $\frac{\alpha}{m} = \frac{.05}{100} = .0005$

# %% [markdown]
# (Other, less straightforward, approaches for adjusting $\alpha$ for multiple comparisons exist. They're beyond our scope, but the major ones are listed under the *General methods of alpha adjustment for multiple comparisons* heading [here](https://en.wikipedia.org/wiki/Multiple_comparisons_problem#See_also).)

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ## Segment 2: Regression

# %% [markdown]
# ### Linear Least Squares for Fitting a Line to Points on a Cartesian Plane

# In [178]
_ = sns.scatterplot(x=x, y=y)

# %% [markdown]
# Consider fitting a line to points on a **Cartesian plane** (2-D surface, with $y$-axis perpendicular to horizontal $x$-axis). To fit such a line, the only parameters we require are a $y$-intercept (say, $\beta_0$) and a slope (say, $\beta_1$):
# 
# $$ y = \beta_0 + \beta_1 x $$
# 
# This corresponds to the case where we have a single feature (a single predictor variable, $x$) in a regression model:
# 
# $$ y = \beta_0 + \beta_1 x + \epsilon $$
# 
# The $\epsilon$ term denotes **error**. For a given instance $i$, $\epsilon_i$ is a measure of the difference between the true $y_i$ and the model's estimate, $\hat{y}_i$. If the model predicts $y_i$ perfectly, then $\epsilon_i = 0$.
# 
# Our objective is to find the parameters $\beta_0$ and $\beta_1$ that minimize $\epsilon$ across all the available data points.
# 
# (Note that sepal length may not be an ideal example of a predictor variable, but these iris data are conveniently available at this stage of the notebook.)

# %% [markdown]
# In the case of a model with a single predictor $x$, there is a fairly straightforward **linear least squares** formula we can use to estimate $\beta_1$:
# $$ \hat{\beta}_1 = \frac{\text{cov}(x,y)}{\sigma^2_x} $$

# %% [markdown]
# (We'll dig further into the "least squares" concept in the next section, for now we can think of it as minimizing the squared error $(\hat{y}_i - y_i)^2$, which we isolate from $\text{cov}(x,y)$ via division by $\sigma^2_x$)

# In [179]
cov

# In [180]
beta1 = cov/np.var(x)
beta1

# %% [markdown]
# With $\hat{\beta}_1$ in hand, we can then rearrange the line equation ($y = \beta_0 + \beta_1 x$) to estimate $\beta_0$:
# $$ \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} $$

# In [181]
beta0 = ybar - beta1*xbar
beta0

# In [182]
xline = np.linspace(4, 8, 1000)
yline = beta0 + beta1*xline

# In [183]
sns.scatterplot(x=x, y=y)
_ = plt.plot(xline, yline, color='orange')

# %% [markdown]
# In regression model terms, if we were provided with a sepal length $x_i$ we could now use the parameter estimates $\hat{\beta}_0$ and $\hat{\beta}_1$ to predict the petal length of an iris:
# $$ \hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i $$

# %% [markdown]
# For instance, our model predicts that an iris with a 5.5cm-long sepal would have 3.1cm-long petal:

# In [184]
x_i = 5.5

# In [185]
y_i = beta0 + beta1*x_i
y_i

# In [186]
sns.scatterplot(x=x, y=y)
plt.plot(xline, yline, color='orange')
_ = plt.scatter(x_i, y_i, marker='o', color='purple')

# %% [markdown]
# As a second example, using the same simulated "Alzheimer's drug" data as the [*Regression in PyTorch* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/regression-in-pytorch.ipynb) and several others in the *ML Foundations* series:

# In [187]
x = np.array([0, 1, 2, 3, 4, 5, 6, 7.])
y = np.array([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])

# In [188]
sns.scatterplot(x=x, y=y)
plt.title("Clinical Trial")
plt.xlabel("Drug dosage (mL)")
_ = plt.ylabel("Forgetfulness")

# In [189]
cov_mat = np.cov(x, y)
cov_mat

# %% [markdown]
# Recalling from above that:
# $$ \hat{\beta}_1 = \frac{\text{cov}(x,y)}{\sigma^2_x} $$

# In [190]
beta1 = cov_mat[0,1]/cov_mat[0,0]
beta1

# %% [markdown]
# ...and that:
# $$ \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} $$

# In [191]
beta0 = y.mean() - beta1*x.mean()
beta0

# %% [markdown]
# ...and, of course, our regression formula:
# $$ \hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i $$

# In [192]
xline = np.linspace(0, 7, 1000)
yline = beta0 + beta1*xline

# %% [markdown]
# By administering 4.5mL of the drug, our model predicts a forgetfulness score of -0.35:

# In [193]
x_i = 4.5

# In [194]
y_i = beta0 + beta1*x_i
y_i

# In [195]
sns.scatterplot(x=x, y=y)
plt.title("Clinical Trial")
plt.xlabel("Drug dosage (mL)")
plt.ylabel("Forgetfulness")
plt.plot(xline, yline, color='orange')
_ = plt.scatter(x_i, y_i, marker='o', color='purple')

# %% [markdown]
# **Exercise**: With data from female Adélie penguins, create a linear least squares model that predicts body mass with flipper length. Predict the mass of a female Adélie penguin that has a flipper length of 197mm.

# In [196]
adelie.head()

# In [197]
x = adelie[adelie.sex == 'Female']['flipper_length_mm'].to_numpy()
y = adelie[adelie.sex == 'Female']['body_mass_g'].to_numpy()/1000

# In [198]
_ = sns.scatterplot(x=x, y=y)

# In [199]
cov_mat = np.cov(x, y)
cov_mat

# In [200]
beta1 = cov_mat[0,1]/cov_mat[0,0]
beta1

# In [201]
beta0 = y.mean() - beta1*x.mean()
beta0

# In [202]
x_i = 197

# In [203]
y_i = beta0 + beta1*x_i
y_i

# In [204]
xline = np.linspace(170, 205, 1000)
yline = beta0 + beta1*xline

# In [205]
sns.scatterplot(x=x, y=y)
plt.title("Female Adélie Penguins")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (kg)")
plt.plot(xline, yline, color='orange')
_ = plt.scatter(x_i, y_i, marker='o', color='purple')

# %% [markdown]
# ### Ordinary Least Squares

# %% [markdown]
# **Ordinary least squares** (OLS) is a linear least squares method we can use to estimate the parameters of regression models that have more than one predictor variable, e.g.:
# 
# $$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \epsilon $$

# %% [markdown]
# Generalizing to $m$ predictors:
# $$ y = \beta_0 + \sum_{j=1}^m \beta_j x_j + \epsilon $$

# %% [markdown]
# To keep the calculations as simple as possible, however, for now we'll stick with a single predictor $x$ (in an example adapted from [here](https://en.wikipedia.org/wiki/Linear_least_squares#Example)):

# In [206]
x = np.array([1, 2, 3, 4.])
y = np.array([6, 5, 7, 10.])

# In [207]
sns.set_style('whitegrid')

# In [208]
fig, ax = plt.subplots()
plt.title('Generative Adversarial Network')
plt.xlabel('Number of convolutional layers')
plt.ylabel('Image realism (out of 10)')
_ = ax.scatter(x, y)

# %% [markdown]
# As is typical in regression model-fitting, we have an *overdetermined* system of linear algebra equations. From the general regression equation $y = \beta_0 + \beta_1 x$, we have four equations (one for each instance $i$) with the two unknown parameters $\beta_0$ and $\beta_1$ shared across the system.
# $$ 6 = \beta_0 + \beta_1 $$
# $$ 5 = \beta_0 + 2\beta_1 $$
# $$ 7 = \beta_0 + 3\beta_1 $$
# $$ 10 = \beta_0 + 4\beta_1 $$

# %% [markdown]
# Since we have more equations than unknowns, we can't solve for the parameters through algebraic rearraging. We can, however, estimate parameters that approximately solve all of the equations with the *Moore-Penrose Pseudoinverse* (from [*Linear Algebra II*](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/2-linear-algebra-ii.ipynb)) or we could use partial-derivative calculus as we'll use here. Either way, with the OLS approach, our objective is to minimize the "sum of squared errors" (SSE).
# 
# The squared error (a.k.a. quadratic cost, from *Calc II*) for a given instance $i$ is $(\hat{y}_i-y_i)^2$.
# 
# The SSE over $n$ instances is then:
# $$ \sum_{i=1}^n (\hat{y}_i-y_i)^2 $$

# %% [markdown]
# In this case, where $\hat{y}_i = \beta_0 + \beta_1 x_i$, we can define the SSE function as:
# $$ S(\beta_0, \beta_1) = \sum_{i=1}^n (\beta_0 + \beta_1 x_i - y_i)^2 $$

# %% [markdown]
# Expanding the summation out over the four instances of $i$:
# $$ S(\beta_0, \beta_1) = [\beta_0 + \beta_1 - 6]^2 + [\beta_0 + 2\beta_1 - 5]^2 + [\beta_0 + 3\beta_1 - 7]^2 + [\beta_0 + 4\beta_1 - 10]^2 $$

# %% [markdown]
# Then (rather laboriously) expanding out the squares and simplifying the result by combining like terms:
# $$ S(\beta_0, \beta_1) = 4\beta_0^2 + 30\beta_1^2 + 20\beta_0\beta_1 - 56\beta_0 - 154\beta_1 + 210 $$

# %% [markdown]
# To minimize SSE, we can now use partial derivatives. Specifically, to find where there is no slope of $S(\beta_0, \beta_1)$ with respect to $\beta_0$:
# $$ \frac{\partial S}{\partial \beta_0} = 8\beta_0 + 20\beta_1 - 56 = 0 $$
# ...and $\beta_1$:
# $$ \frac{\partial S}{\partial \beta_1} = 20\beta_0 + 60\beta_1 - 154 = 0 $$

# %% [markdown]
# Rearranging, we obtain a system of two linear equations called the **normal equations** (however many parameters are in the model is how many rows of equations we'll have in the system):
# $$ 8\beta_0 + 20\beta_1 = 56 $$
# $$ 20\beta_0 + 60\beta_1 = 154 $$

# %% [markdown]
# A handy numerical approach for solving for $\beta_0$ and $\beta_1$ is matrix inversion (which we covered in detail toward the end of the [*Intro to Linear Algebra* notebook](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/1-intro-to-linear-algebra.ipynb)).

# %% [markdown]
# To avoid confusion with with the broader $x$ (number of GAN conv layers) and $y$ variables (image realism), let's use $A$ for the matrix of "inputs" and $z$ for the vector of "outputs", with the vector $w$ containing the unknown weights $\beta_0$ and $\beta_1$:
# $$ Aw = z $$

# In [209]
A = np.array([[8, 20],[20, 60]])
A

# In [210]
z = np.array([56, 154])

# %% [markdown]
# To solve for $w$, we can invert $A$ (assuming $A$ is not singular; i.e., all of its columns are independent):
# $$ w = A^{-1}z $$

# In [211]
Ainv = np.linalg.inv(A)
Ainv

# In [212]
w = np.dot(Ainv, z)
w

# %% [markdown]
# Thus, the line that minimizes the squared error across all four equations has the parameters $\hat{\beta}_0 = 3.5$ and $\hat{\beta}_1 = 1.4$:
# $$ \hat{y} = 3.5 + 1.4 x $$

# In [213]
xline = np.linspace(1, 4, 1000)
yline = w[0] + w[1]*xline

# In [214]
fig, ax = plt.subplots()
plt.title('Generative Adversarial Network')
plt.xlabel('Number of convolutional layers')
plt.ylabel('Image realism (out of 10)')
ax.scatter(x, y)
_ = plt.plot(xline, yline, color='orange')

# %% [markdown]
# For fun, following the linear algebra in the slides, we could output $\hat{y}$ across all the instances $i$:
# $$ \hat{y} = Xw $$

# In [215]
X = np.concatenate([np.matrix(np.ones(x.size)).T, np.matrix(x).T], axis=1)
X

# In [216]
yhat = np.dot(X, w)
yhat

# %% [markdown]
# Incidentally, **residuals** are the distances between $\hat{y}_i$ and $y_i$:

# In [217]
fig, ax = plt.subplots()
plt.title('Generative Adversarial Network')
plt.xlabel('Number of convolutional layers')
plt.ylabel('Image realism (out of 10)')
ax.scatter(x, y)
_ = plt.plot(xline, yline, color='orange')
for i in range(x.size):
    plt.plot([x[i],x[i]], [y[i],yhat[0,i]], color='darkred')

# %% [markdown]
# The square of these residuals is what we minimize with SSE in OLS regression.

# %% [markdown]
# The above OLS approach expands to a wide variety of circumstances:
# 
# * Multiple features ($x$, the predictors)
# * Polynomial (typically quadratic) features, e.g., $y = \beta_0 + \beta_1 x + \beta_2 x^2$
# * Interacting features, e.g., $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2$
# * Discrete, categorical features, incl. any combination of continuous and discrete features

# %% [markdown]
# As an example of the latter...

# In [218]
iris

# In [219]
sns.set_style('darkgrid')

# In [220]
_ = sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris)

# In [221]
import pandas as pd

# In [222]
dummy = pd.get_dummies(iris.species, dtype = 'int64') # int64 dtype required to avoid ValueError when calling sm.OLS() below
dummy

# In [223]
y = iris.petal_length

# In [224]
X = pd.concat([iris.sepal_length, dummy.setosa, dummy.versicolor], axis=1)
X # virginia as "baseline" where setosa and versicolor are both 0

# In [225]
import statsmodels.api as sm

# In [226]
X = sm.add_constant(X)
X

# In [227]
model = sm.OLS(y, X)

# In [228]
result = model.fit()

# In [229]
result.summary()

# %% [markdown]
# Our earlier iris model, with sepal length as the only predictor of petal length, had $r^2 = 0.76$. In our latest iris model, a whopping 97% of the variance in petal length is explained by the predictors.

# In [230]
beta = result.params
beta

# In [231]
xline = np.linspace(4, 8, 1000)
vi_yline = beta['const'] + beta['sepal_length']*xline
se_yline = beta['const'] + beta['sepal_length']*xline + beta['setosa']
ve_yline = beta['const'] + beta['sepal_length']*xline + beta['versicolor']

# In [232]
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris)
plt.plot(xline, vi_yline, color='darkgreen')
plt.plot(xline, se_yline, color='darkblue')
_ = plt.plot(xline, ve_yline, color='orange')

# %% [markdown]
# Now using our refined model, such that it predicts the petal length of a *versicolor* iris with a 5.5cm-long sepal:

# In [233]
x_sepall_i = 5.5
x_setosa_i = 0
x_versic_i = 1

# In [234]
y_i = beta['const'] + beta['sepal_length']*x_sepall_i + beta['setosa']*x_setosa_i + beta['versicolor']*x_versic_i

# In [235]
y_i

# In [236]
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris)
plt.plot(xline, vi_yline, color='darkgreen')
plt.plot(xline, se_yline, color='darkblue')
plt.plot(xline, ve_yline, color='orange')
_ = plt.scatter(x_sepall_i, y_i, marker='o', color='purple')

# %% [markdown]
# ...or a *virginica* with a sepal of the same length:

# In [237]
x_sepall_i = 5.5
x_setosa_i = 0
x_versic_i = 0 # the only change

# In [238]
x_i = np.array([1, x_sepall_i, x_versic_i, x_versic_i])

# In [239]
y_i = np.dot(beta, x_i)

# In [240]
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris)
plt.plot(xline, vi_yline, color='darkgreen')
plt.plot(xline, se_yline, color='darkblue')
plt.plot(xline, ve_yline, color='orange')
_ = plt.scatter(x_i[1], y_i, marker='o', color='purple')

# %% [markdown]
# (It is also possible to have the slope vary by categorical variable, not only the $y$-intercept. This is a *hierarchical linear model* and the classic text on it is [Gelman & Hill, 2006](https://amzn.to/3hoOevb).)

# %% [markdown]
# **Exercise**:
# Download the California housing dataset (process is immediately below) and use the statsmodels `OLS()` method to create a model that uses at least a few of the provided features to predict house price.

# In [241]
from sklearn.datasets import fetch_california_housing

# In [242]
housing = fetch_california_housing()

# In [243]
# Median house price in a given "census block group" in $100,000s (from the 1990 U.S. Census):
y = housing.target
y[0:20]

# In [244]
# There are 20k block groups in the dataset...
len(y)

# In [245]
# ...distributed all across California:
lat = housing.data[:, housing.feature_names.index('Latitude')]
lon = housing.data[:, housing.feature_names.index('Longitude')]

# Create the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(lon, lat, alpha=0.5, s=5, c=housing.target, cmap='viridis')

plt.title('California Housing Prices Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label('Median House Value (100k$)')

# Annotate major cities (approximate coordinates)
cities = {
    'San Francisco': (-122.4194, 37.7749),
    'Los Angeles': (-118.2437, 34.0522),
    'San Diego': (-117.1611, 32.7157),
    'Sacramento': (-121.4944, 38.5816),
    'Fresno': (-119.7871, 36.7378)
}

for city, coords in cities.items():
    plt.annotate(city, xy=coords, xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

# In [246]
X = pd.DataFrame(housing.data)
X.columns = housing.feature_names
X

# In [247]
# Description of the feature variables (and more on the dataset in general):
print(housing.DESCR)

# In [247]

# In [247]

# In [247]

# In [247]

# %% [markdown]
# ### Logistic Regression

# %% [markdown]
# Reasonably often we'd like to have a regression model that predicts a binary outcome (e.g., identifying if a fast-food item is a hot dog or not a hot dog). This can be accomplished with **logistic regression**, which adapts linear regression by including the *logit* function:
# $$ x = \text{log}(\frac{p}{1-p}) $$
# This function uses the natural logarithm and maps a binary probability $p$ (which can only range from zero to one) to an unbounded range ($-\infty$ to $\infty$).

# In [248]
def logit(my_p): # this is also available as scipy.special.logit()
    return np.log(my_p/(1-my_p))

# In [249]
logit(0.5)

# In [250]
logit(0.1)

# In [251]
logit(0.01) # closer to zero approaches negative infinity

# In [252]
logit(0.99) # closer to one approaches positive infinity

# %% [markdown]
# More specifically, logistic regression makes use of the *expit* function (a.k.a., logistic function), which is the inverse of the logit. That is, it returns a probability $p$ when passed some unbounded input $x$:
# $$ p = \frac{1}{1+e^{-x}} $$

# In [253]
def expit(my_x): # this is also available as scipy.special.expit()
    return 1/(1+np.exp(-my_x))

# In [254]
expit(4.59512)

# %% [markdown]
# This logistic function allows us to map the unbounded output of a linear regression model to a probability ranging from zero to one.

# %% [markdown]
# Let's dig right into a hands-on example:

# In [255]
titanic = sns.load_dataset('titanic')

# In [256]
titanic

# In [257]
np.unique(titanic['survived'], return_counts=True)

# In [258]
np.unique(titanic['sex'], return_counts=True)

# In [259]
np.unique(titanic['class'], return_counts=True)

# In [260]
_ = sns.displot(titanic['age'], kde=True)

# In [261]
gender = pd.get_dummies(titanic['sex'], dtype = 'int64')
gender

# In [262]
clas = pd.get_dummies(titanic['class'], dtype = 'int64')
clas

# In [263]
y = titanic.survived

# In [264]
X = pd.concat([clas.First, clas.Second, gender.female, titanic.age], axis=1)
X = sm.add_constant(X)
X

# In [265]
model = sm.Logit(y, X, missing='drop') # some rows contain NaN

# In [266]
result = model.fit()

# In [267]
result.summary()

# In [268]
beta = result.params
beta

# %% [markdown]
# As an example, our model suggests a 17-year-old female traveling in first class (such as Rose in the 1997 James Cameron film) had a 95.9% of chance of surviving:

# In [269]
linear_out = beta['const'] + beta['First']*1 + beta['Second']*0 + beta['female']*1 + beta['age']*17
linear_out

# In [270]
expit(linear_out)

# %% [markdown]
# In contrast, a 20-year-old male traveling in third class had an 11.2% chance of surviving:

# In [271]
jack = np.array([1, 0, 0, 0, 20])

# In [272]
linear_out = np.dot(beta, jack)
linear_out

# In [273]
expit(linear_out)

# %% [markdown]
# **Exercises**
# 
# 1. Use the scikit-learn `train_test_split()` method ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)) to split the titanic data into a training data set (say, two thirds of the data) and a test data set (one third of the data).
# 
# 2. Re-train the OLS model above using your newly-created training data set. Using the test data set, test the model's quality, e.g., with respect to:
# 
#     * Accuracy (using a standard 50% binary classification threshold)
#     * Area under the receiving operator characteristic curve (we covered this in [Calculus II](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/4-calculus-ii.ipynb)).
# 
# 3. Use your own creative whims to add additional features to an OLS model and train it using the training set. How does your new model compare on the test data set metrics relative to the baseline model? And how do they compare to the [Kaggle leaderboard](https://www.kaggle.com/c/titanic/leaderboard)?

# %% [markdown]
# **Return to slides here.**

# %% [markdown]
# ## Segment 3: Bayesian Statistics

# %% [markdown]
# ### Bayes' Theorem

# %% [markdown]
# ...allows us to find $P(\text{x}|\text{y})$ when we have $P(\text{y}|\text{x})$:
# $$ P(\text{x}|\text{y}) = \frac{P(\text{x})P(\text{y}|\text{x})}{P(\text{y})} $$

# %% [markdown]
# Let's use the *xkcd* [exploding sun cartoon](https://xkcd.com/1132/) as an example. Using a standard $\alpha = .05$ threshold, the frequentist rejected the null hypothesis that the sun hadn't exploded because the probability the neutrino detector outputs `YES` when the sun hasn't exploded is $\frac{1}{6} \times \frac{1}{6} = \frac{1}{36} \approx 0.0278$, which is $<.05$.

# %% [markdown]
# Using Bayes' theorem, the Bayesian statistician incorporates additional information -- largely related to the probability that the sun has exploded irrespective what the neutrino detector says -- to draw a different conclusion.
# 
# Let's likewise use some back-of-the-envelope figures to estimate the probability the sun has exploded ($x = \text{exploded}$) given the neutrino detector output `YES` ($y = \text{YES}$); that is, $P(x|y)$. To find this, we'll need $P(y|x)$, $P(x)$, and $P(y)$.
# 
# $P(y = \text{YES} | x = \text{exploded}) = \frac{35}{36} \approx 0.972 $.
# 
# $P(x = \text{exploded})$ can be roughly estimated. It is generous to assume a $\frac{1}{1000}$ chance because for every thousand days that pass, the sun explodes far less often than once. Further, unless the sun had exploded only in the past few minutes, we'd already be dead. So, the probability that we are alive in an instant where the sun has exploded is extremely small. Anyway, let's go with $\frac{1}{1000}$ because even with this exceedingly generous figure, we'll demonstrate the point.
# 
# $P(y = \text{YES})$: As is often the case, this probability in the Bayes' theorem denominator can be calculated with information we already have because:
# $$ P(\text{y}) = \sum_x P(\text{y}|x)P(x) $$
# Summing over the two possible states of x ($x =$ exploded, $x =$ not exploded):
# $$ P(y = \text{YES}) = P(\text{YES given exploded})P(\text{exploded}) + P(\text{YES given not exploded})P(\text{not exploded}) $$
# $$ = \left(\frac{35}{36}\right)\left(\frac{1}{1000}\right) + \left(\frac{1}{36}\right)\left(\frac{999}{1000}\right) $$
# ...which comes out to $P(y = \text{YES}) \approx 0.0287$:

# In [274]
py = (35/36.)*(1/1000.) + (1/36.)*(999/1000.)
py

# %% [markdown]
# Now we have everything we need to apply Bayes' theorem:

# In [275]
py_givenx = 0.972
px = .0001

# In [276]
def bayes(my_px, my_pygivenx, my_py):
    return (my_px*my_pygivenx)/my_py

# In [277]
bayes(px, py_givenx, py)

# %% [markdown]
# Therefore, even with our generous baseline estimate of a $\frac{1}{1000}$ chance of the sun having exploded, Bayes' rule enables us to find a 0.3% chance the sun has exploded given the neutrino detector output `YES`. Certainly odds that merit making a $50 bet!

# %% [markdown]
# It would require several hours to describe Bayesian inference beyond Bayes' theorem, but here are "getting started" pages for each of the primary Python libraries, ordered roughly from most lightweight (easier to pick up) to most involved:
# 
# * [NumPyro](https://num.pyro.ai/en/latest/getting_started.html)
# * [PyMC](https://www.pymc.io/projects/docs/en/stable/learn.html)
# * [PyStan](https://pystan.readthedocs.io/en/latest/)

# %% [markdown]
# **Return to slides here.**

