#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10000)


# In[8]:


data = pd.read_csv('3D_spatial_network.txt', sep=",", header=None)


# In[9]:


y=data.iloc[:,3]
X=data.iloc[:,1:3]


# In[45]:


def normalize(X):
    return (X-X.mean(axis=0))/(X.std(axis=0))


# In[46]:


X=normalize(X)


# In[47]:


y=normalize(y)


# In[48]:


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


# In[49]:


cv=CrossValidation(kind='holdout')
train_X,train_y,test_X,test_y=cv.train_test_split(X,y,train_per=80,randomize=False)


# In[139]:


class Polynomial_Regression:
    def __init__(self,degree=3,alpha=0.01,reg_factor=0.5,stop_rate=1e-8,stop_criteria='cost'):
        self.degree=degree
        self.alpha=alpha
        self.reg_factor=reg_factor
        self.loaded=False
        self.stop_rate= stop_rate
        self.stop_criteria=stop_criteria
        self.cost=[]
    def normalize(self,X):
        return (X-X.mean(axis=0))/X.std(axis=0) 
    def combination_generator(self,degree=3):
        l=[(i,j) for i in range(degree+1) for j in range(degree+1) if i or j is not 0 if i+j<=degree]
        return l
    def transform(self,X,degree=3):
        self.degree=degree
        X=self.normalize(X)
        features=np.ones((X.shape[0],1))
        self.combs=self.combination_generator(self.degree)
        for i in range(len(self.combs)):
            comb_feature=(np.power(X.iloc[:,0].values,self.combs[i][0])*np.power(X.iloc[:,1].values,self.combs[i][1])).flatten()
            features=np.append(features,comb_feature.reshape(-1,1),axis=1)
        return features
    
    def train(self,X,y,max_iter=100,norm='l2',verbose=True):
        if self.degree:
            X=self.transform(X)
        else:
            X=X.values
        y=self.normalize(y).values.reshape(-1,1)
        if self.loaded==False:
            self.w=np.random.randn(X.shape[1],1)
        for i in range(max_iter):
            m=X.shape[0]
            y_pred=X@self.w
            if norm=='l2':
                grad=1/m*(X.T@(y_pred - y))+self.reg_factor*self.w
                cost=1/2/m*(sum(y-y_pred)**2 + self.reg_factor*sum(self.w**2))
            elif norm=='l1':
                grad=1/m*(X.T@(y_pred - y))+self.reg_factor*np.sign(self.w)
                cost=1/2/m*(sum(y-y_pred)**2) + self.reg_factor*sum(np.abs(self.w))
            self.w-=self.alpha*grad
            self.cost.append(cost)
            if self.stop_criteria=='cost' and len(self.cost)>1 and abs(self.cost[i-1]-self.cost[i])<=self.stop_rate :
                print('Early Stopping at iteration',i,' cost ',self.cost[i],'using criteria ',self.stop_criteria,self.cost[i-1])
                self.final_iter=i
                break
            elif self.stop_criteria=='slope' and np.linalg.norm(grad*self.alpha)<=self.stop_rate:
                print('Early Stopping at iteration',i,' cost ',self.cost[i],'using criteria ',self.stop_criteria)
                self.final_iter=i
                break
                
            if verbose and i%10==0:
                print("Iteration {}: Cost: {}".format(i,cost[0]), end="\r")

    def predict(self,X):
        X=self.transform(X)
        self.dummy=X   
        y_pred=X@self.w
        return y_pred
    
    def load_weights(self,w):
        self.w=w
        self.loaded=True


# In[81]:


pol1=Polynomial_Regression(degree=1,alpha=0.1,reg_factor=0)
pol1.train(train_X,train_y,max_iter=501,norm='l2');
plt.plot(pol1.cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0,100])


# In[82]:


pol2=Polynomial_Regression(degree=2,alpha=0.1,reg_factor=0)
pol2.train(train_X,train_y,max_iter=501,norm='l2');
plt.plot(pol2.cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0,100])


# In[83]:


pol3=Polynomial_Regression(degree=3,alpha=0.1,reg_factor=0)
pol3.train(train_X,train_y,max_iter=501,norm='l2');
plt.plot(pol3.cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0,100])


# In[84]:


pol4=Polynomial_Regression(degree=4,alpha=0.1,reg_factor=0)
pol4.train(train_X,train_y,max_iter=501,norm='l2');
plt.plot(pol4.cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0,100])


# In[142]:


pol4_reg=Polynomial_Regression(degree=4,alpha=0.05,reg_factor=0.1)
pol4_reg.train(train_X,train_y,max_iter=501,norm='l2');
plt.plot(pol4_reg.cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0,100])


# In[144]:


r2_score(pol4_reg.predict(test_X),test_y.values.reshape(-1,1)),rmse(pol4_reg.predict(test_X),test_y.values.reshape(-1,1))


# In[ ]:


pol2_reg=Polynomial_Regression(degree=4,alpha=0.1,reg_factor=0.1)
pol2_reg.train(train_X,train_y,max_iter=501,norm='l2');
plt.plot(pol2_reg.cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0,100])


# In[ ]:


def r2_score(y_pred,y):
    y_bar=y.mean()  # Mean of actual y
    sum_of_squares_total=sum((y-y_bar)**2)
    sum_of_squares_reg=sum((y_pred-y_bar)**2)
    sum_of_squares_res=sum((y_pred-y)**2)

    r2_score=1-sum_of_squares_res/sum_of_squares_total
    return r2_score


# In[ ]:


def rmse(y_pred,y):
    return np.sqrt(((y_pred - y) ** 2).mean())


# In[ ]:


r2_score(pol.predict(test_X),test_y.values.reshape(-1,1))


# In[ ]:


rmse(pol.predict(test_X),test_y.values.reshape(-1,1))


# In[85]:


r2_score(pol3.predict(test_X),test_y.values.reshape(-1,1)),rmse(pol3.predict(test_X),test_y.values.reshape(-1,1))


# In[86]:


r2_score(pol4.predict(test_X),test_y.values.reshape(-1,1)),rmse(pol4.predict(test_X),test_y.values.reshape(-1,1))


# In[58]:


y_pred=pol.predict(train_X)


# In[59]:


for i in range(100):
    plt.scatter(i,y_pred[i],c='red',label='predicted')
for i in range(100):
    plt.scatter(i,pol.normalize(train_y)[i],c='blue',label='actual')
# plt.legend()
plt.show()


# In[128]:


pol6_no_reg=Polynomial_Regression(degree=6,alpha=0.1,reg_factor=0)
pol6_no_reg.train(train_X,train_y,max_iter=501,norm='l2');
plt.plot(pol6_no_reg.cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0,100])


# In[129]:


r2_score(pol6_no_reg.predict(test_X),test_y.values.reshape(-1,1)),rmse(pol6_no_reg.predict(test_X),test_y.values.reshape(-1,1))


# In[95]:


pol6_l2=Polynomial_Regression(degree=6,alpha=0.1,reg_factor=0.1)
pol6_l2.train(train_X,train_y,max_iter=501,norm='l2');
plt.plot(pol6_l2.cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0,100])


# In[104]:


r2_score(pol6_l2.predict(test_X),test_y.values.reshape(-1,1)),rmse(pol6_l2.predict(test_X),test_y.values.reshape(-1,1))


# In[130]:


pol6_l1=Polynomial_Regression(degree=6,alpha=0.01,reg_factor=0.1)
pol6_l1.train(train_X,train_y,max_iter=501,norm='l1');
plt.plot(pol6_l1.cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0,100])


# In[115]:


r2_score(pol6_l1.predict(test_X),test_y.values.reshape(-1,1)),rmse(pol6_l1.predict(test_X),test_y.values.reshape(-1,1))


# In[ ]:




