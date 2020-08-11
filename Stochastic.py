#!/usr/bin/env python
# coding: utf-8

# In[246]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import math
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[48]:


data = pd.read_csv('3D_spatial_network.txt', sep=",", header=None)


# In[49]:


X=data.iloc[:,1:-1]
y=data.iloc[:,-1]


# In[50]:


def normalize(X):
    return (X-X.mean(axis=0))/(X.std(axis=0))


# In[51]:


X=normalize(X)


# In[53]:


y=normalize(y)


# In[290]:


class CrossValidation:
    def __init__(self,kind='holdout',seed=2010):
        self.kind=kind
        np.random.seed(seed)
        
    def train_test_split(self,X,y,train_per=False,n_folds=False,randomize=True):
        if self.kind=='holdout':
            if randomize==True:
                train_rows=random.sample(range(0,y.size),train_per*y.size//100)
                train_rows.sort()
                test_rows=[rows for rows in X.index.values if rows not in train_rows]
                return X.iloc[train_rows].reset_index(drop=True),y.iloc[train_rows].reset_index(drop=True),X.iloc[test_rows].reset_index(drop=True),y.iloc[test_rows].reset_index(drop=True)
            else:
                return X.iloc[:train_per*y.size//100,:].reset_index(drop=True),y.iloc[:train_per*y.size//100].reset_index(drop=True),X.iloc[train_per*y.size//100:y.size,:].reset_index(drop=True),y.iloc[train_per*y.size//100:y.size].reset_index(drop=True)
        
        elif self.kind=='fold':
            
            total=random.sample(range(0,y.size),y.size)
            train_rows=[]
            test_rows=[]
            for i in range(n_folds):
                test_selected_row=total[y.size//n_folds*i:y.size//n_folds*(i+1)]
                test_rows.append(test_selected_row)
                test_rows[i].sort()
                train_rows.append([row for row in total if row not in test_selected_row])
                train_rows[i].sort()
            return train_rows,test_rows


# In[291]:


cv=CrossValidation(kind='holdout')
train_X,train_y,test_X,test_y=cv.train_test_split(X,y,train_per=80,randomize=False)


# In[292]:


class stochastic_gradient_descent:
    def __init__(self,alpha,no_iterations,reg_factor,stop_rate=1e-8,stop_criteria='cost'):
        self.alpha = alpha
        self.no_iterations = no_iterations
        self.reg_factor=reg_factor
        self.cost=[]
        self.w1=[]
        self.w2=[]
        self.stop_rate= stop_rate
        self.stop_criteria=stop_criteria
        self.final_iter=no_iterations
        
    def fit(self,X,y,verbose=True,seed=2019):
        self.w=np.random.rand(X.shape[1],1)
        np.random.seed(seed)
        for i in range(self.no_iterations): 
            m=y.size
            r = np.random.randint(0,m-1)
            y_pred=X.iloc[r]@self.w
            grad=self.reg_factor*self.w+((y_pred - y[r])*X.iloc[r]).values.reshape(-1,1)
            if self.stop_criteria=='slope' and np.linalg.norm(grad*self.alpha)<=self.stop_rate:
                print('Early Stopping at iteration',i,' cost ',self.cost[i],'using criteria ',self.stop_criteria)
                self.final_iter=i
                break
            self.w -= self.alpha*(grad)
            cost=1/2/m*(np.sum((X@self.w-y.to_numpy().reshape(-1,1))**2))+self.reg_factor*sum(self.w**2)
            self.cost.append(cost.to_numpy()[0])
            if self.stop_criteria=='cost' and len(self.cost)>1 and abs(self.cost[i-1]-self.cost[i])<=self.stop_rate :
                print('Early Stopping at iteration',i,' cost ',self.cost[i],'using criteria ',self.stop_criteria, 'pervious cost:', self.cost[i-1])
                self.final_iter=i
                break
            if verbose and i%100==0:
                print("loss {} at iteration {}".format(cost.to_numpy()[0],i))
        return self
    def predict(self,X):
        return X@self.w


# In[281]:


linear_stoc_grad=stochastic_gradient_descent(alpha=0.01,no_iterations=5000,reg_factor=0)
linear_stoc_grad=linear_stoc_grad.fit(train_X,train_y,verbose=True)


# In[282]:


linear_stoc_grad.w


# In[284]:


n_iterations=len(linear_stoc_grad.cost)//20
indexes=[i*20 for i in range(n_iterations)]
indexes.append(linear_stoc_grad.final_iter-1)
plt.plot(indexes, np.array(linear_stoc_grad.cost)[indexes])
plt.xlabel('iteration')
plt.ylabel('Cost')


# In[285]:


plt.plot(np.array(linear_stoc_grad.cost))
plt.xlabel('iteration')
plt.ylabel('Cost')


# In[286]:


def r2_score(y_pred,y):
    y_bar=y.mean()  # Mean of actual y
    sum_of_squares_total=sum((y-y_bar)**2)
    sum_of_squares_reg=sum((y_pred-y_bar)**2)
    sum_of_squares_res=sum((y_pred-y)**2)

    r2_score=1-sum_of_squares_res/sum_of_squares_total
    return r2_score


# In[287]:


def rmse(y_pred,y):
    return np.sqrt(((y_pred - y) ** 2).mean())


# In[288]:


r2_score(linear_stoc_grad.predict(test_X).values,test_y.values.reshape(-1,1))


# In[289]:


rmse(linear_stoc_grad.predict(test_X).values,test_y.values.reshape(-1,1))


# In[ ]:




