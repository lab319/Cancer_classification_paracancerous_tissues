# -*- coding: utf-8 -*-
"""
Created on Mon Dec 02 15:56:29 2019

@author: Administrator
"""

import numpy as np
from sklearn import metrics    
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy import interp
import openpyxl

'''
best_params
y_pred
y_proba
y_score
fpr
tpr

'''


file_1=open("KIRC_methy.txt")
file_1.readline()
x=np.loadtxt(file_1)
file_1.close()

file_4=open("KIRC_label.txt")
file_4.readline()
y=np.loadtxt(file_4)
file_4.close()

scores_1=[]
scores_2=[]
scores_3=[]
scores_4=[]
scores_5=[]
scores_6=[]
scores_7=[]
scores_8=[]
scores_9=[]
fprs=[]
tprs=[]
mean_fpr=np.linspace(0,1,100)
pre_values=[]
pre_labels=[]
i=1
kf = KFold(n_splits=5, shuffle=True, random_state=5)
for train_index,test_index in kf.split(x, y):
   print("i",i)
   x_train,x_test=x[train_index],x[test_index]
   y_train,y_test=y[train_index],y[test_index]
   
   ss=StandardScaler()
   x_train=ss.fit_transform(x_train)
   x_test=ss.transform(x_test)
   
   param_grid = [{'kernel': ['rbf'],'C': [3,4,5,7],'gamma':[1e-7,1e-6,1e-5]}]
   clf = SVC(probability=True,random_state=3)
   
   kfold = KFold(n_splits=3)
   grid = GridSearchCV(clf, param_grid, cv=kfold, scoring='roc_auc',n_jobs=-1,verbose=1)
   grid.fit(x_train, y_train)
   score_train=grid.best_score_
   print ("grid.best_score_",score_train)
   print ("grid.best_params_",grid.best_params_)
   
   best_params=grid.best_params_
   model=SVC(**best_params,probability=True,random_state=3)
   
   model.fit(x_train, y_train)  
   y_proba=model.predict_proba(x_test)[:,1]
   y_pred = model.predict(x_test)
   y_score = model.decision_function(x_test)
   
   auc_score = metrics.roc_auc_score(y_test, y_proba) 
   acc_score = accuracy_score(y_test, y_pred)
   print(auc_score)
   print(acc_score)
    
   precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_proba)
   pr_auc = metrics.auc(recall, precision)
   mcc = matthews_corrcoef(y_test, y_pred)
    
   tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
   total=tn+fp+fn+tp
   sen = float(tp)/float(tp+fn)
   sps = float(tn)/float((tn+fp))
   
   fpr,tpr,threshold = roc_curve(y_test, y_proba)
   
   p = precision_score(y_test, y_pred)
   r = recall_score(y_test, y_pred)
   f1 = f1_score(y_test,y_pred)
  
    
   scores_1.append(auc_score)
   scores_2.append(acc_score)
   scores_3.append(pr_auc)
   scores_4.append(mcc)
   scores_5.append(sen)
   scores_6.append(sps)
   scores_7.append(p)
   scores_8.append(r)
   scores_9.append(f1)
   
   pre_values.append(y_proba)
   pre_labels.append(y_pred)
   
   fprs.append(fpr)
   tprs.append(interp(mean_fpr,fpr,tpr))
   tprs[-1][0]=0.0
   i = i+1

df = pd.DataFrame(tprs)
df.to_excel('svm-kirc-methy-tprs.xlsx',index=False,header=False,float_format='%.10f')
  
print('auc-mean-score: %.3f' %np.mean(scores_1))
print('acc-mean-score: %.3f' %np.mean(scores_2))
print('pr-mean-score: %.3f' %np.mean(scores_3))
print('mcc-mean-score: %.3f' %np.mean(scores_4))
print('sen-mean-score: %.3f' %np.mean(scores_5))
print('sps-mean-score: %.3f' %np.mean(scores_6))
print('precision-mean-score: %.3f' %np.mean(scores_7))
print('recall-mean-score: %.3f' %np.mean(scores_8))
print('f1-mean-score: %.3f' %np.mean(scores_9))

def write_excel_csv(path, sheet_name, value):
    index = len(value)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.cell(row=i+1, column=j+1, value=str(value[i][j]))
    workbook.save(path)
    print("写入数据成功！")



prelabels_name_csv = 'svm-KIRC-methy-labels.csv'
sheet_name2_csv = 'svm-KIRC-methy-labels'


write_excel_csv(prelabels_name_csv, sheet_name2_csv,pre_labels)

'''
auc-mean-score: 0.764
acc-mean-score: 0.650
pr-mean-score: 0.827
mcc-mean-score: 0.298
sen-mean-score: 0.750
sps-mean-score: 0.552
precision-mean-score: 0.700
recall-mean-score: 0.750
f1-mean-score: 0.703

'''
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
     
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   