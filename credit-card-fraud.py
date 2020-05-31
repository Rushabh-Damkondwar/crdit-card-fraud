#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import os
import matplotlib.pyplot as plt
import math
pd.set_option('display.max_rows', None) #To display all rows
pd.set_option('display.max_columns', None) # To display all columns
from datetime import date
from datetime import  time
import seaborn as sns


# In[5]:


cf=pd.read_csv("C:/Users/Rushabh/Downloads/Data science software/Python projects/Credit card fraud/creditcard.csv")


# In[6]:


cf.head()


# In[7]:


cf.isnull().sum()


# # Data visulizaation 

# In[8]:


#By using this graph we can see the clear data imbalance.With help of this we can decide our further 
#action and the implementation of the algorithm.
print("Distribuition of Normal(0) and Frauds(1): ")
print(cf["Class"].value_counts())

plt.figure(figsize=(7,5))
sns.countplot(cf['Class'])
plt.title("Class Count", fontsize=18)
plt.xlabel("Is fraud?", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


# In[9]:


timedelta = pd.to_timedelta(cf['Time'], unit='s')
cf['Time_min'] = (timedelta.dt.components.minutes).astype(int)
cf['Time_hour'] = (timedelta.dt.components.hours).astype(int)


# In[10]:


cf.tail(200)


# In[11]:


sns.violinplot(cf['Class'], cf['Amount']) #Variable Plot
sns.despine()


# In[12]:


#We can see the std distribution and the outliers in the data.

def plot_comparison(x, title):
    fig, ax = plt.subplots(3, 1, sharex=True)
    sns.distplot(x, ax=ax[0])
    ax[0].set_title('Histogram + KDE')
    sns.boxplot(x, ax=ax[1])
    print()
    ax[1].set_title('Boxplot')
    print()
    sns.violinplot(x, ax=ax[2])
    print()
    ax[2].set_title('Violin plot')
    print("\n")
    fig.suptitle(title, fontsize=16)
    print()
    


# In[13]:


plt.show()


# In[14]:


sample_gaussian = np.random.normal(size=50000)
plot_comparison(sample_gaussian, 'Standard Normal Distribution')


# In[15]:



    plt.scatter(cf['Class'], cf['Amount'])


# In[16]:


cf['Amount'].max()


# In[17]:


plt.hist(cf['Amount'], bins=15)


# In[18]:


#Exploring the distribuition by Class types throught hours and minutes
plt.figure(figsize=(12,5))
sns.distplot(cf[cf['Class'] == 0]["Time_hour"], 
             color='g')
sns.distplot(cf[cf['Class'] == 1]["Time_hour"], 
             color='r')
plt.title('Fraud x Normal Transactions by Hours', fontsize=17)
plt.xlim([-1,25])
plt.show()


# In[19]:


#Exploring the distribuition by Class types throught hours and minutes
plt.figure(figsize=(12,5))
sns.distplot(cf[cf['Class'] == 0]["Time_min"], 
             color='g')
sns.distplot(cf[cf['Class'] == 1]["Time_min"], 
             color='r')
plt.title('Fraud x Normal Transactions by minutes', fontsize=17)
plt.xlim([-1,61])
plt.show()


# In[20]:


#To clearly the data of frauds and no frauds
df_fraud = cf[cf['Class'] == 1]
df_normal = cf[cf['Class'] == 0]

print("Fraud transaction statistics")
print(cf["Amount"].describe())
print("\nNormal transaction statistics")
print(cf["Amount"].describe())


# In[21]:


cf.head()


# In[22]:


#Looking the Amount and time distribuition of FRAUD transactions
ax = sns.lmplot(y="Amount", x="Time_min", fit_reg=False,aspect=1.8,
                data=cf, hue='Class')
plt.title("Amounts by Minutes of Frauds and Normal Transactions",fontsize=16)
#plt.show()


# In[23]:


#In the below diagram we can see the data distribution of each and every feature and we can decide 
# and find out the fraudulant data nature.It also helps you to decide the signifricant features.

import matplotlib.gridspec as gridspec
#Looking the V's features
columns = cf.iloc[:,1:29].columns

frauds = cf.Class == 1
normals = cf.Class == 0

grid = gridspec.GridSpec(14, 2)
plt.figure(figsize=(15,20*4))

for n, col in enumerate(cf[columns]):
    ax = plt.subplot(grid[n])
    sns.distplot(cf[col][frauds], bins = 50, color='g') #Will receive the "semi-salmon" violin
    sns.distplot(cf[col][normals], bins = 50, color='r') #Will receive the "ocean" color
    ax.set_ylabel('Density')
    ax.set_title(str(col))
    ax.set_xlabel('')
plt.show()


# In[24]:


cf = cf[["Time_hour","Time_min","V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19","V27","Amount","Class"]]


# In[25]:



cf.Amount = np.log(cf.Amount + 0.001)


# In[26]:


cf.head()


# In[ ]:


# Here we are distributing the data and implementing the different algorithm.Also we are checking for the accuracy and ROC curve where we will be able to decide the which is good and giving the right accuracy.


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[28]:


x=cf.iloc[:,0:16]
y=cf.iloc[:,-1]


# In[29]:


x_train1,x_test,y_train1,y_test=train_test_split(x,y,test_size=.3,random_state=100)


# In[30]:


x_train,y_train= SMOTE().fit_sample(x_train1,y_train1)


# In[31]:


#Showing the diference before and after the transformation used
print("normal data distribution: {}".format(Counter(y)))
x_smote,y_smote = SMOTE().fit_sample(x,y)
print("SMOTE data distribution: {}".format(Counter(y_smote)))


# In[32]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve, accuracy_score


# # Random forest algorithm

# In[54]:



classifier=RandomForestClassifier(n_estimators=80)
classifier.fit(x_train,y_train)


# In[55]:


pred=classifier.predict(x_test)
print("Your prediction is:", pred)


# In[56]:


pred


# In[57]:


from sklearn.metrics import confusion_matrix
tab1 = confusion_matrix(pred,y_test)
tab1


# In[58]:


tab1.diagonal().sum() / tab1.sum() *100


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = classifier.predict_proba(x_test)[:,1]

# Generate precision recall curve values: precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()


# # Logistic regression

# In[31]:


logistic=LogisticRegression()
logistic.fit(x_train,y_train)
log_pred=logistic.predict(x_test)


# In[32]:


print("Logistic prediction is:",log_pred)


# In[36]:


tab2=confusion_matrix(log_pred,y_test)
print("logistic confusion matrix is:")
tab2


# In[35]:


tab2.diagonal().sum() / tab2.sum() *100


# In[34]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logistic.predict_proba(x_test)[:,1]

# Generate precision recall curve values: precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()


# # Decision Tree

# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
tree_pred=tree.predict(x_test)
print("Prediction of Decision tree is:")
tree_pred


# In[38]:


from sklearn.metrics import confusion_matrix
tab3= confusion_matrix(tree_pred,y_test)
tab3.diagonal().sum() / tab3.sum() *100


# In[39]:


print("Accuracy of Decision tree is:")
tab3


# In[40]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = tree.predict_proba(x_test)[:,1]

# Generate precision recall curve values: precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()


# In[ ]:





# In[45]:


#Adaboosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.01, 0.1, 1, 10],
             'penalty':['l1', 'l2']}

#LogisticRegression(random_state=2)


# In[46]:



#grid=GridSearchCV(RandomForestClassifier(max_features=3, max_depth=2,n_estimators=10,criterion='entropy', n_jobs=1, verbose=1),param_grid=param_grid,scoring="recall")
#adaboosting=AdaBoostClassifier(LogisticRegression(),n_estimators=200)
#grid=GridSearchCV(LogisticRegression(random_state=2),param_grid =param_grid,scoring="recall")
adaboosting=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=200)
adaboosting.fit(x_train,y_train)
Ada_pred= adaboosting.predict(x_test)


# In[47]:


Ada_pred


# In[48]:


from sklearn.metrics import confusion_matrix
tab2 = confusion_matrix(Ada_pred,y_test)
tab2


# In[49]:


tab2.diagonal().sum() / tab2.sum() *100


# In[50]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = tree.predict_proba(x_test)[:,1]

# Generate precision recall curve values: precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()


# In[52]:


from sklearn.metrics import classification_report
print(classification_report(Ada_pred,y_test))


# In[ ]:





# In[61]:


x_train.shape


# In[62]:


y_train.shape


# In[65]:


y_test.shape


# In[66]:


x_test.shape


# # Conclusion:

#  After applying the different algorithms and checking the acuracy we got to know tha random forest algorithm
# plays  a greate job.By using the Adaboosting and GridSearch CV algorithm we were able to see some auuracy.

# In[ ]:




