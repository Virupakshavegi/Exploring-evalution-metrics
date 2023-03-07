#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
d=pd.read_csv("eighthr.csv")


# In[81]:


d.head()


# In[82]:


d.isnull().sum()


# In[83]:


cs=d.shape[1]
print(cs)


# In[84]:


cr=d.shape[0]
print(cr)


# In[85]:


from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values='?', strategy='constant',fill_value=0.0)
d=imp_mean.fit_transform(d)
d=pd.DataFrame(d)
d.head()


# In[86]:


x=d.iloc[:,1:73]
y=d.iloc[:,[73]].astype('int')
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
dt=DecisionTreeClassifier()
l=LogisticRegression()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
l=l.fit(x_train,y_train)
dt=dt.fit(x_train,y_train)
#y_pred=l.predict(x_test)


# In[87]:


y_pred=l.predict(x_test)
from sklearn.metrics import confusion_matrix,roc_auc_score
matrix = confusion_matrix(y_test, y_pred)
# Accuracy
#from sklearn.metrics import accuracy_score
acc = (matrix[0,0]+matrix[1,1])/(matrix[0,0]+matrix[0,1]+matrix[1,0]+matrix[1,1])
# Recall
#from sklearn.metrics import recall_score
#rc=recall_score(y_test, y_pred)
# Precision
#from sklearn.metrics import precision_score
#pc=precision_score(y_test, y_pred)
p = matrix[0,0]/(matrix[0,0]+matrix[1,0])
r = matrix[0,0]/(matrix[0,0]+matrix[1,1])
f = matrix[0,0]/(matrix[0,0]+0.5*(matrix[1,0]+matrix[1,1]))
roc=roc_auc_score(y_test, y_pred)
print(acc)
print(p)
print(r)
print(f)
print(roc)


# In[88]:


y_pred=dt.predict(x_test)
matrix = confusion_matrix(y_test, y_pred)
# Accuracy
#from sklearn.metrics import accuracy_score
acc = (matrix[0,0]+matrix[1,1])/(matrix[0,0]+matrix[0,1]+matrix[1,0]+matrix[1,1])
# Recall
#from sklearn.metrics import recall_score
p = matrix[0,0]/(matrix[0,0]+matrix[1,0])
r = matrix[0,0]/(matrix[0,0]+matrix[1,1])
f = matrix[0,0]/(matrix[0,0]+0.5*(matrix[1,0]+matrix[1,1]))
roc=roc_auc_score(y_test, y_pred)
print(acc)
print(p)
print(r)
print(f)
print(roc)


# In[ ]:





# In[ ]:




