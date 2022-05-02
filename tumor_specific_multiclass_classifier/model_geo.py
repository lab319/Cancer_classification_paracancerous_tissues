# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 13:52:11 2021

@author: Lab319
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from sklearn.metrics import accuracy_score, roc_auc_score


tcgadata = pd.read_table("multiclass_input_data/tcgadata.txt")
tcgalabel = pd.read_table("multiclass_input_data/tcgalabel.txt").values.reshape([-1, ])

geodata = pd.read_table("multiclass_input_data/geo.txt")
geolabel = pd.read_table("multiclass_input_data/geo_label.txt").values.reshape([-1, ])

n_classes = 3
geo_one_hot = label_binarize(geolabel, np.arange(3))

params={#'objective': 'multi:softmax',
        'objective': 'multi:softmax',
                   'num_class':3,
                   'booster':'gbtree',
                   'seed':12345}
  
    
clf = XGBClassifier(**params)
model = clf.fit(tcgadata.iloc[:,1:6], tcgalabel)
#y_pre = model.predict_proba(geodata.iloc[:,1:6]).reshape(geolabel.shape[0], 3)
y_pre = model.predict(geodata.iloc[:,1:6])
#ylabel = np.argmax(y_pre, axis=1)
acc = accuracy_score(geolabel,y_pre)
acc_kirc = accuracy_score(geolabel[0:45],y_pre[0:45])
acc_brca = accuracy_score(geolabel[46:85],y_pre[46:85])
acc_thca = accuracy_score(geolabel[86:126],y_pre[86:126])
print('ACC : %f' % acc)
print('ACC_kirc : %f' % acc_kirc)
print('ACC_brca : %f' % acc_brca)
print('ACC_thca : %f' % acc_thca)

