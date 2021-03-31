#!/usr/bin/env python
# coding: utf-8

# In[1]:


# classification - decision tree implemetaiton
# aim : to classify the iris plant species given in the dataset using the decision tree algorithm.

import numpy as np                # linear algebra
import pandas as pd               # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


# importing the dataset and seeing it's shape
df = pd.read_csv('Iris.csv')
df.shape


# In[3]:


df.head()


# In[4]:


X = df.drop(['Species','Id'], axis=1)
y = df['Species']
X.head()


# In[5]:


from category_encoders import OrdinalEncoder

encoder = OrdinalEncoder()
encoder.fit(X)
X_enc = encoder.transform(X)
X_enc.head()


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, train_size = .66)
X_train.shape


# In[7]:


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)
model = classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)


# In[8]:


classifier.score(X_test, y_test)


# In[9]:


model.classes_


# In[10]:


model.feature_importances_


# In[11]:


list(zip(X.columns, model.feature_importances_))


# In[16]:


import matplotlib.pyplot as plt # data visualization

plt.figure()
plt.title("Feature importances")
plt.barh(X.columns, model.feature_importances_, 1)


# In[17]:


from sklearn import tree
import matplotlib.pyplot as plt # data visualization

plt.figure(figsize=(20,10))

tree.plot_tree(model, feature_names = X.columns, class_names = model.classes_, label='root') 


# In[18]:


from sklearn.metrics import confusion_matrix

y_predict_test = classifier.predict(X_test)
confusion_matrix(y_test, y_predict_test)


# In[19]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict_test))


# In[ ]:




