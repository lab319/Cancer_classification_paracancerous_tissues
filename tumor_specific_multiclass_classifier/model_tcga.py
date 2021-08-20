
# coding: utf-8

# In[12]:


import xgboost
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import sklearn.metrics 
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,Binarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from xgboost.sklearn import XGBClassifier


data = pd.read_table("E:/cbj/three-diseases/classification/multiclassification/second/tcgadata.txt")
data.columns = [str(i) for i in range(data.shape[1])]
label = pd.read_table("E:/cbj/three-diseases/classification/multiclassification/second/tcgalabel.txt").values.reshape([-1, ])

X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.2,random_state = 123)
    
params={'objective': 'multi:softmax',
            #'objective': 'multi:softprob',
                   'num_class':3,
                   'booster':'gbtree',
                   'seed':12345}
  
    
clf = XGBClassifier(**params)
model = clf.fit(X_train, y_train)
#y_pre = model.predict_proba(test_data).reshape(test_y.shape[0], 3)
y_pre = model.predict(X_test)
acc = accuracy_score(y_test,y_pre)
print ('ACC : %f' % acc) 




    


    

























