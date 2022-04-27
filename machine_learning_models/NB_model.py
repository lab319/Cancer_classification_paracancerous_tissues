# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:27:06 2019

@author: Lab319
"""

import numpy as np
from sklearn import metrics   
from sklearn.preprocessing import StandardScaler  
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import precision_score, recall_score,f1_score
from scipy import interp
from sklearn.metrics import roc_curve, auc
import openpyxl

file_1=open("data.txt")
file_1.readline()
x=np.loadtxt(file_1)
file_1.close()

file_4=open("label.txt")
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
   
   model= GaussianNB()
   
   model.fit(x_train, y_train)  
   y_proba=model.predict_proba(x_test)[:,1]
   y_pred = model.predict(x_test)
    
   fpr,tpr,threshold = roc_curve(y_test, y_proba) ###计算真正率和假正率
   roc_auc = auc(fpr,tpr) ###计算auc的值
   print('roc_auc',roc_auc)
   
   auc_score = metrics.roc_auc_score(y_test, y_proba) 
   acc_score = accuracy_score(y_test, y_pred)
   
   print("auc_score",auc_score)
    
   precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_proba)
   pr_auc = metrics.auc(recall, precision)
   mcc = matthews_corrcoef(y_test, y_pred)
    
   tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
   total=tn+fp+fn+tp
   sen = float(tp)/float(tp+fn)
   sps = float(tn)/float((tn+fp))
   
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
df.to_excel('nb-kirc-methy-tprs.xlsx',index=False,header=False,float_format='%.10f')
  
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



prelabels_name_csv = 'nb-KIRC-methy-labels.csv'
sheet_name2_csv = 'nb-KIRC-methy-labels'


write_excel_csv(prelabels_name_csv, sheet_name2_csv,pre_labels)



'''
auc-mean-score: 0.674
acc-mean-score: 0.656
pr-mean-score: 0.795
mcc-mean-score: 0.350
sen-mean-score: 0.631
sps-mean-score: 0.717
precision-mean-score: 0.747
recall-mean-score: 0.631
f1-mean-score: 0.668

'''



