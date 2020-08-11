#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[3]:


data = pd.read_csv('3D_spatial_network.txt', sep=",", header=None)


# In[4]:


X=data.iloc[:,1:-1]
y=data.iloc[:,-1]


# In[5]:


def normalize(X):
    return (X-X.mean(axis=0))/(X.std(axis=0))


# In[6]:


X=normalize(X)


# In[7]:


y=normalize(y)


# In[12]:


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


# In[13]:


cv=CrossValidation(kind='holdout')
train_X,train_y,test_X,test_y=cv.train_test_split(X,y,train_per=80,randomize=False)


# In[39]:


class linear_gradient_descent:
    def __init__(self,alpha,no_iterations,reg_factor,stop_rate,stop_criteria):
        self.alpha = alpha
        self.no_iterations = no_iterations
        self.reg_factor=reg_factor
        self.cost=[]
        self.w1=[]
        self.w2=[]
        self.stop_rate= stop_rate
        self.stop_criteria=stop_criteria
        self.final_iter=no_iterations
    def fit(self,X,y,reg_type='l2',reg_factor=0.05,verbose=True,seed=2019):
        self.w=np.random.rand(X.shape[1],1)
        np.random.seed(seed)
        for i in range(self.no_iterations):
            m=y.size
            y_pred=X@self.w
            if reg_type=='l2':
                grad=X.T@((y_pred - y.to_numpy().reshape(-1,1)))+self.reg_factor*self.w
            elif reg_type=='l1':
                grad=X.T@((y_pred - y.to_numpy().reshape(-1,1)))+self.reg_factor/2*np.sign(self.w)
            self.w = self.w -(1/m)*self.alpha*(grad)
            cost=1/2/m*(np.sum((X@self.w-y.to_numpy().reshape(-1,1))**2))+self.reg_factor*sum(self.w**2)
            self.cost.append(list(cost.to_numpy())[0])
            
            if self.stop_criteria=='cost' and len(self.cost)>1 and self.cost[i-1]-self.cost[i]<=self.stop_rate :
                print('Early Stopping at iteration',i,' cost ',self.cost[i],'using criteria ',self.stop_criteria)
                self.final_iter=i
                break
            elif self.stop_criteria=='slope' and np.linalg.norm(grad*self.alpha)<=self.stop_rate:
                print('Early Stopping at iteration',i,' cost ',self.cost[i],'using criteria ',self.stop_criteria)
                self.final_iter=i
                break
            self.w1.append(self.w.iloc[0])
            self.w2.append(self.w.iloc[1])
            if verbose and i%100==0:
                print("loss {} at iteration {}".format(cost.to_numpy()[0],i))
        return self
    def predict(self,X):
        return X@self.w


# In[49]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=0.1,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[50]:


linear_grad.w


# In[51]:


linear_grad.cost


# In[57]:


n_iterations=len(linear_grad.cost)//20
indexes=[i*20 for i in range(n_iterations)]
indexes.append(linear_grad.final_iter-1)
plt.plot(indexes, np.array(linear_grad.cost)[indexes])
plt.xlabel('iteration')
plt.ylabel('Cost')


# In[58]:


def r2_score(y_pred,y):
    y_bar=y.mean()  # Mean of actual y
    sum_of_squares_total=sum((y-y_bar)**2)
    sum_of_squares_reg=sum((y_pred-y_bar)**2)
    sum_of_squares_res=sum((y_pred-y)**2)

    r2_score=1-sum_of_squares_res/sum_of_squares_total
    return r2_score


# In[59]:


def rmse(y_pred,y):
    return np.sqrt(((y_pred - y) ** 2).mean())


# In[60]:


r2_score(linear_grad.predict(test_X).values,test_y.values.reshape(-1,1))


# In[61]:


rmse(linear_grad.predict(test_X).values,test_y.values.reshape(-1,1))


# In[104]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-20,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_1=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[105]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-15,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_2=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[106]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-10,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_3=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[92]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-7,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_4=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[93]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-6,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_5=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[94]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-5,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_6=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[95]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-4,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_7=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[96]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-3,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_8=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[97]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-2,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_9=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[98]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-1,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_10=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l1')


# In[107]:


reg_predictions_r2=[]
reg_predictions_rmse=[]


# In[108]:


reg_predictions_r2.append(r2_score(linear_grad_1.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2.append(r2_score(linear_grad_2.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2.append(r2_score(linear_grad_3.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2.append(r2_score(linear_grad_4.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2.append(r2_score(linear_grad_5.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2.append(r2_score(linear_grad_6.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2.append(r2_score(linear_grad_7.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2.append(r2_score(linear_grad_8.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2.append(r2_score(linear_grad_9.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2.append(r2_score(linear_grad_10.predict(test_X).values,test_y.values.reshape(-1,1)))


# In[109]:


reg_predictions_rmse.append(rmse(linear_grad_1.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse.append(rmse(linear_grad_2.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse.append(rmse(linear_grad_3.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse.append(rmse(linear_grad_4.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse.append(rmse(linear_grad_5.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse.append(rmse(linear_grad_6.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse.append(rmse(linear_grad_7.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse.append(rmse(linear_grad_8.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse.append(rmse(linear_grad_9.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse.append(rmse(linear_grad_10.predict(test_X).values,test_y.values.reshape(-1,1)))


# In[110]:


reg_predictions_r2,reg_predictions_rmse


# In[117]:


reg_factors=[]


# In[118]:


reg_factors.append(linear_grad_1.reg_factor)
reg_factors.append(linear_grad_2.reg_factor)
reg_factors.append(linear_grad_3.reg_factor)
reg_factors.append(linear_grad_4.reg_factor)
reg_factors.append(linear_grad_5.reg_factor)
reg_factors.append(linear_grad_6.reg_factor)
reg_factors.append(linear_grad_7.reg_factor)
reg_factors.append(linear_grad_8.reg_factor)
reg_factors.append(linear_grad_9.reg_factor)
reg_factors.append(linear_grad_10.reg_factor)


# In[132]:


plt.plot(np.log10(reg_factors),reg_predictions_rmse)


# In[119]:


reg_factors


# In[131]:


np.log10(reg_factors,)


# In[133]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-20,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_1=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[134]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-15,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_2=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[135]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-10,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_3=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[136]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-7,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_4=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[137]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-6,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_5=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[138]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-5,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_6=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[139]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-4,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_7=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[140]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-3,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_8=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[141]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-2,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_9=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[142]:


linear_grad=linear_gradient_descent(alpha=0.01,reg_factor=1e-1,no_iterations=5000,stop_rate=1e-7,stop_criteria='cost')
linear_grad_10=linear_grad.fit(train_X,train_y,verbose=True,reg_type='l2')


# In[144]:


reg_predictions_r2_l2=[]
reg_predictions_rmse_l2=[]


# In[145]:


reg_predictions_r2_l2.append(r2_score(linear_grad_1.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2_l2.append(r2_score(linear_grad_2.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2_l2.append(r2_score(linear_grad_3.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2_l2.append(r2_score(linear_grad_4.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2_l2.append(r2_score(linear_grad_5.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2_l2.append(r2_score(linear_grad_6.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2_l2.append(r2_score(linear_grad_7.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2_l2.append(r2_score(linear_grad_8.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2_l2.append(r2_score(linear_grad_9.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_r2_l2.append(r2_score(linear_grad_10.predict(test_X).values,test_y.values.reshape(-1,1)))


# In[148]:


reg_predictions_rmse_l2.append(rmse(linear_grad_1.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse_l2.append(rmse(linear_grad_2.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse_l2.append(rmse(linear_grad_3.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse_l2.append(rmse(linear_grad_4.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse_l2.append(rmse(linear_grad_5.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse_l2.append(rmse(linear_grad_6.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse_l2.append(rmse(linear_grad_7.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse_l2.append(rmse(linear_grad_8.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse_l2.append(rmse(linear_grad_9.predict(test_X).values,test_y.values.reshape(-1,1)))
reg_predictions_rmse_l2.append(rmse(linear_grad_10.predict(test_X).values,test_y.values.reshape(-1,1)))


# In[162]:


reg_predictions_r2_l2,reg_predictions_rmse_l2


# In[151]:


reg_factors_l2=[]


# In[152]:


reg_factors_l2.append(linear_grad_1.reg_factor)
reg_factors_l2.append(linear_grad_2.reg_factor)
reg_factors_l2.append(linear_grad_3.reg_factor)
reg_factors_l2.append(linear_grad_4.reg_factor)
reg_factors_l2.append(linear_grad_5.reg_factor)
reg_factors_l2.append(linear_grad_6.reg_factor)
reg_factors_l2.append(linear_grad_7.reg_factor)
reg_factors_l2.append(linear_grad_8.reg_factor)
reg_factors_l2.append(linear_grad_9.reg_factor)
reg_factors_l2.append(linear_grad_10.reg_factor)


# In[153]:


plt.plot(np.log10(reg_factors_l2),reg_predictions_rmse_l2)


# In[ ]:




