#!/usr/bin/env python
# coding: utf-8

# In[203]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import math
from scipy.special import gamma
from scipy.stats import beta


# In[204]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10000)


# In[205]:


X=np.append(np.random.uniform(0,0.5,30),np.random.uniform(0.5,1,130)).reshape(-1,1)


# In[206]:


plt.scatter(range(160),X)
plt.xlabel('Points')
plt.ylabel('Probability')


# In[207]:


X.mean()


# In[208]:


def likelihood_func(X,univ_mean=None):
    if univ_mean:
        mean=univ_mean
    else:
        mean=X.mean(axis=0)
    bern_matrix=np.power(mean,X)*np.power(1-mean,1-X)
    return np.prod(bern_matrix,axis=0)


# In[209]:


def normalize(X):
    return (X-X.mean(axis=0))/X.std(axis=0)


# In[210]:


def beta_func(X=None,mean=None,prev_a=None,prev_b=None,update=False):
    a=prev_a
    b=prev_b
    if update:
        a+=len(X[X>=0.5])
        b+=len(X[X<0.5])
        return a,b
    else:
        pass
    gamma_a_b=gamma(a+b)/(gamma(a)*gamma(b))
    prior_mu=np.power(mean,a-1)*np.power(1-mean,b-1)*gamma_a_b
    return a,b,prior_mu


# In[219]:


iterations_value=[]
likelihood_list=[]
beta_list=[]
a=np.random.rand(1)
b=3/2*a
iteration_list=[x/100 for x in range(100)]
for i in range(len(X)):
    posterior=[]
    l=[]
    bb=[]
    for j in range(1,101):
        mu=j/100
        likelihood=likelihood_func(X[i],univ_mean=mu)
        a,b,beta=beta_func(prev_a=a,prev_b=b,mean=mu)
        l.append(likelihood)
        bb.append(beta)
        posterior.append(beta*likelihood)
    iterations_value.append(posterior)
    likelihood_list.append(l)
    beta_list.append(bb)
    plt.plot(iteration_list,beta_list[i])
    plt.title('Coin '+str(i+1))
    plt.xlabel('Mean')
    plt.savefig('beta'+str(i+1)+'.png')
    plt.show()
    print(a,b)
    a,b=beta_func(X[i],prev_a=a,prev_b=b,mean=mu,update=True)


# In[47]:


likelihood_func(X,X.mean())


# In[231]:


a=np.random.rand(1)
b=3/2*a
gamma_a_b=gamma(a+b)/(gamma(a)*gamma(b))
mean=X.mean()
prior_mu=np.power(mean,a-1)*np.power(1-mean,b-1)*gamma_a_b
prior_mu
posterior=prior_mu*likelihood_func(X,X.mean())
print(posterior,prior_mu)


# In[222]:


a


# In[153]:


a=np.random.rand(1)
b=3/2*a
iterations_value=[]
likelihood_list=[]
beta_list=[]
posterior=[]
l=[]
bb=[]
iteration_list=[x/100 for x in range(100)]
for j in range(1,101):
    mu=j/100
    likelihood=likelihood_func(X,univ_mean=mu)
    a,b,beta=beta_func(prev_a=a,prev_b=b,mean=mu)
    l.append(likelihood)
    bb.append(beta)
    posterior.append(beta*likelihood)


# In[154]:


plt.plot(iteration_list,bb)
plt.title('Prior')
plt.xlabel('Mean')
plt.show()
# Prior Plot


# In[155]:


plt.plot(iteration_list,l)
plt.title('Likelihood')
plt.xlabel('Mean')
# Likelihood Plot


# In[156]:


plt.plot(iteration_list,posterior)
plt.xlabel('Mean')
plt.title('Posterior')
# Posterior Plot


# In[ ]:





# In[188]:


X_C=np.random.uniform(0,1,160).reshape(-1,1)


# In[189]:


X_C.mean()


# In[218]:


plt.scatter(range(160),X_C)
plt.xlabel('Points')
plt.ylabel('Probability')


# In[141]:


X_D=np.random.uniform(0,1,200).reshape(-1,1)


# In[142]:


X_D.mean()


# In[191]:


a=np.random.rand(1)
b=3/2*a
iterations_value=[]
likelihood_list=[]
beta_list=[]
posterior=[]
l=[]
bb=[]
iteration_list=[x/100 for x in range(100)]
for j in range(1,101):
    mu=j/100
    likelihood=likelihood_func(X_C,univ_mean=mu)
    a,b,beta=beta_func(prev_a=a,prev_b=b,mean=mu)
    l.append(likelihood)
    bb.append(beta)
    posterior.append(beta*likelihood)


# In[192]:


plt.plot(iteration_list,bb)
plt.title('Prior')
plt.xlabel('Mean')
plt.show()
# Prior Plot


# In[193]:


plt.plot(iteration_list,l)
plt.title('Likelihood')
plt.xlabel('Mean')
# Likelihood Plot


# In[194]:


plt.plot(iteration_list,posterior)
plt.title('Posterior')
plt.xlabel('Mean')
# Posterior Plot


# In[195]:


a=np.random.rand(1)
b=3/2*a
iterations_value=[]
likelihood_list=[]
beta_list=[]
posterior=[]
l=[]
bb=[]
iteration_list=[x/100 for x in range(100)]
for j in range(1,101):
    mu=j/100
    likelihood=likelihood_func(X_D,univ_mean=mu)
    a,b,beta=beta_func(prev_a=a,prev_b=b,mean=mu)
    l.append(likelihood)
    bb.append(beta)
    posterior.append(beta*likelihood)


# In[196]:


plt.plot(iteration_list,bb)
plt.title('Prior')
plt.xlabel('Mean')
plt.show()
# Prior Plot


# In[197]:


plt.plot(iteration_list,l)
plt.title('Likelihood')
plt.xlabel('Mean')
# Likelihood Plot


# In[198]:


plt.plot(iteration_list,posterior)
plt.title('Posterior')
plt.xlabel('Mean')
# Posterior Plot


# In[ ]:




