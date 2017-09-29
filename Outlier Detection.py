
# coding: utf-8

# # Outlier Detection Algorithm - Isolation Forests
# 
# ## BNP Paribas Kaggle Data Set
# 
# Data source: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
# 
# 
# Outlier Detection- Ensemble unsupervised learning method - Isolation Forest
# 
# The isolation algorithm is an unsupervised machine learning method used to detect abnormal anomalies in data such as outliers. This is once again a randomized & recursive partition of the training data in a tree structure. The number of sub samples and tree size is specified and tuned appropriately. The distance to the outlier is averaged calculating an anomaly detection score: 1 = outlier 0 = close to zero are normal data.
# 

# ### 1. Load Data & Libraries

# In[1]:

#Load the data and make dataset a Pandas DataFrame
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
df = pd.read_csv("C:\\data\\claims.csv")
df.head(1)


# In[2]:

print("Number of missing values: {0}".format ((df.shape[0] * df.shape[1])-df.count().sum()))


# In[3]:

#Missing Value PVI using mean value of attributes;
x = df.fillna(df.mean())
x.head(2)


# ### 2. Label Encoding

# In[4]:

#Label encoder tranforms any label or attribute for input to the algorithim 
#we can also see some missing values in the top few rows of the data set these will also
#need to be treated in a suitable mannor.
for feature in df.columns:
    if df[feature].dtype=='object':
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
df.tail(3)


# ### 3. Split Data to Training/Test Set

# In[5]:

X = df.drop(['ID'],['target'], axis = 1)
Y = df.target.values
#Cross - Validation - split the data into 70% training and the remainder for testing the model
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)


# ### 4. Build Isolation Forest

# In[ ]:

#Build isolation forests!
from sklearn.ensemble import IsolationForest
rs = np.random.RandomState(30)
#Build the ensemble model
isomodel = IsolationForest(max_samples = 256, random_state=rs)
isomodelfit = isomodel.fit(X_train)#fit the ensemble model and build the trees
y_pred_train = isomodel.predict(X_train)
y_pred_test = isomodel.predict(X_test)#find average depth
y_pred_test


# In[ ]:

#plot the results of the isolation forest
import plotly.plotly as py
import plotly.graph_objs as go
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = isomodel.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Isolation Forest Outlier Detection")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
b2 = plt.scatter(X_train[:, 0], X_train[:, 1], c='green')
c = plt.scatter(y_pred_outliers[:, 0], y_pred_outliers[:, 1], c='red')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()


# In[ ]:



