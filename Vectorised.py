#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[4]:


data = pd.read_csv('3D_spatial_network.txt', sep=",", header=None)


# In[5]:


X=data.iloc[:,1:-1]
y=data.iloc[:,-1]


# In[6]:


def normalize(X):
    return (X-X.mean(axis=0))/(X.std(axis=0))


# In[7]:


X=normalize(X)


# In[8]:


y=normalize(y)


# In[9]:


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


# In[10]:


cv=CrossValidation(kind='holdout')
train_X,train_y,test_X,test_y=cv.train_test_split(X,y,train_per=80,randomize=False)


# In[44]:


class vectorized_linear_gradient_descent:
    def __init__(self,reg_factor=0.5):
        self.reg_factor=reg_factor
    def fit(self,X,y):
        I=np.identity(X.shape[1])
        self.w=np.linalg.inv(X.T@X+self.reg_factor*I)@X.T@y
        m=y.size
        self.w=self.w.values.reshape(-1,1)
        return self
    def predict(self,X):
        return X@self.w


# In[45]:


vect_grad=vectorized_linear_gradient_descent(reg_factor=0)
vect_grad=vect_grad.fit(train_X,train_y)


# In[46]:


def r2_score(y_pred,y):
    y_bar=y.mean()  # Mean of actual y
    sum_of_squares_total=sum((y-y_bar)**2)
    sum_of_squares_reg=sum((y_pred-y_bar)**2)
    sum_of_squares_res=sum((y_pred-y)**2)

    r2_score=1-sum_of_squares_res/sum_of_squares_total
    return r2_score


# In[47]:


def rmse(y_pred,y):
    return np.sqrt(((y_pred - y) ** 2).mean())


# In[48]:


r2_score(vect_grad.predict(test_X).values,test_y.values.reshape(-1,1))


# In[50]:


rmse(vect_grad.predict(test_X).values,test_y.values.reshape(-1,1))


# In[ ]:




