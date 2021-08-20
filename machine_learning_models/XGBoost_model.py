
# coding: utf-8

# In[12]:


import xgboost
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import sklearn.metrics 

class GV:
    '''
    Scoring	Function	   Comment
    *Classification
    ‘accuracy’             metrics.accuracy_score
    ‘average_precision’	   metrics.average_precision_score
    ‘f1’	               metrics.f1_score	for binary targets
    ‘f1_micro’	           metrics.f1_score	micro-averaged
    ‘f1_macro’         	   metrics.f1_score	macro-averaged
    ‘f1_weighted’	       metrics.f1_score	weighted average
    ‘f1_samples’	       metrics.f1_score	by multilabel sample
    ‘neg_log_loss’	       metrics.log_loss	requires predict_proba support
    ‘precision’ etc.	   metrics.precision_score	suffixes apply as with ‘f1’
    ‘recall’ etc.	       metrics.recall_score	suffixes apply as with ‘f1’
    ‘roc_auc’	           metrics.roc_auc_score

    *Clustering
    ‘adjusted_rand_score’	metrics.adjusted_rand_score

    *Regression
    ‘neg_mean_absolute_error’	metrics.mean_absolute_error
    ‘neg_mean_squared_error’	metrics.mean_squared_error
    ‘neg_median_absolute_error’	metrics.median_absolute_error
    ‘r2’	metrics.r2_score
    '''

    def xg_find_base(self, scoring, data_x, data_y, model_xg, params, overfit=None):
        kfold = KFold(n_splits=3)
        params = {}
        
        params_test1 = {"max_depth": np.arange(3, 8, 1),"min_child_weight":np.arange(1, 10, 1)}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'max_depth': clf.best_params_["max_depth"]})
        params.update({'min_child_weight': clf.best_params_["min_child_weight"]})
        model_xg.max_depth = clf.best_params_["max_depth"]
        model_xg.min_child_weight = clf.best_params_["min_child_weight"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)
        
        params_test1 = {"learning_rate": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.13]}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'learning_rate': clf.best_params_["learning_rate"]})
        model_xg.learning_rate = clf.best_params_["learning_rate"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)
        
        params_test1 = {"colsample_bytree": np.arange(0.1, 1, 0.1), 'subsample': np.arange(0.1, 0.9, 0.1)}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'colsample_bytree': clf.best_params_["colsample_bytree"]})
        params.update({'subsample': clf.best_params_["subsample"]})
        model_xg.colsample_bytree = clf.best_params_["colsample_bytree"]
        model_xg.subsample = clf.best_params_["subsample"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)
        
        params_test1 = {"gamma": np.arange(0.1, 2, 0.1)}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'gamma': clf.best_params_["gamma"]})
        model_xg.gamma = clf.best_params_["gamma"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)
        
        params_test1 = {'reg_lambda': np.arange(0.5, 1.6, 0.1), 'reg_alpha': np.arange(0, 0.7, 0.1)}
        clf = GridSearchCV(model_xg, params_test1, cv=kfold, n_jobs=-1, scoring=scoring)
        clf.fit(data_x, data_y)
        params.update({'reg_lambda': clf.best_params_["reg_lambda"]})
        params.update({'reg_alpha': clf.best_params_["reg_alpha"]})
        model_xg.reg_lambda = clf.best_params_["reg_lambda"]
        model_xg.reg_alpha = clf.best_params_["reg_alpha"]
        print(clf.best_params_)
        print("clf.best_score_", clf.best_score_)
        return model_xg, params
        
        
        
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,Binarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import roc_curve, auc
import openpyxl
from scipy import interp

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
params_all=[]
pre_values=[]
pre_labels=[]

data = pd.read_table("KIRC_methy.txt")
data.columns = [str(i) for i in range(data.shape[1])]

label = pd.read_table("KIRC_label.txt").values.reshape([-1, ])

kf = KFold(n_splits=5, shuffle=True, random_state=5)
y_valid_pred_total = np.zeros(data.shape[0])
score = []
print(data.shape, label.shape)
for train_ind, test_ind in kf.split(data, label):
    train_data = data.iloc[train_ind, :]
    train_y = label[train_ind]
    test_data = data.iloc[test_ind, :]
    test_y = label[test_ind]
    model = XGBClassifier()
    gv = GV()
    params = {}

    model, params = gv.xg_find_base('roc_auc', train_data, train_y, model, {})
    #     model,params = gv.xg_find_up('roc_auc',train_data,train_y,model,{},overfit=True)
    print(params)
    early_stop = 50
    verbose_eval = 0
    num_rounds = 450

#     train_data,evals_data,train_y,evals_y = train_test_split(train_data,train_y,test_size=0.2)
    d_train = xgb.DMatrix(train_data, label=train_y)
    d_valid = xgb.DMatrix(test_data, label=test_y)
#     d_evals = xgb.DMatrix(evals_data, label=evals_y)

    watchlist = [(d_train, 'train')]
    params.update({'eval_metric': 'auc',
                   'objective': 'binary:logistic',
                   'booster':'gbtree',
                   'seed':1})
    model = xgb.train(params, d_train, num_boost_round=num_rounds, early_stopping_rounds=early_stop, evals=watchlist)

    y_pre = model.predict(d_valid).reshape([-1,1])
    mms = Binarizer(0.5)
    y_pre_ = mms.fit_transform(y_pre)
    auc_score = metrics.roc_auc_score(test_y, y_pre)
    acc_score = accuracy_score(test_y, y_pre_)
    
    precision, recall, _thresholds = metrics.precision_recall_curve(test_y, y_pre)
    pr_auc = metrics.auc(recall, precision)
    mcc = matthews_corrcoef(test_y, y_pre_)
    
    tn, fp, fn, tp = confusion_matrix(test_y, y_pre_).ravel()
    total=tn+fp+fn+tp
    sen = float(tp)/float(tp+fn)
    sps = float(tn)/float((tn+fp))
    
    fpr,tpr,threshold = roc_curve(test_y, y_pre) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    
    p = precision_score(test_y, y_pre_)
    r = recall_score(test_y, y_pre_)
    f1 = f1_score(test_y,y_pre_)
    
    print ('AUC : %f' % auc_score)
    print ('ACC : %f' % acc_score) 
    print("PRAUC: %f" % pr_auc)
    print ('MCC : %f' % mcc)
    print ('SEN : %f' % sen)
    print ('SEP : %f' % sps)
    print ('P : %f' % p)
    print ('R : %f' % r)
    print ('F1 : %f' % f1)
    
    
    scores_1.append(auc_score)
    scores_2.append(acc_score)
    scores_3.append(pr_auc)
    scores_4.append(mcc)
    scores_5.append(sen)
    scores_6.append(sps)
    scores_7.append(p)
    scores_8.append(r)
    scores_9.append(f1)
    
    params_all.append(params)
    pre_values.append(y_pre)
    pre_labels.append(y_pre_)
    
    fprs.append(fpr)
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0


#df = pd.DataFrame(tprs)
#df.to_excel('xgb-kirc-methy-tprs.xlsx',index=False,header=False,float_format='%.10f')

    
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



prelabels_name_csv = 'xgb-KIRC-methy-labels.csv'
sheet_name2_csv = 'xgb-KIRC-methy-labels'


write_excel_csv(prelabels_name_csv, sheet_name2_csv,pre_labels)



"""

auc-mean-score: 0.780
acc-mean-score: 0.675
pr-mean-score: 0.842
mcc-mean-score: 0.353
sen-mean-score: 0.747
sps-mean-score: 0.597
precision-mean-score: 0.703
recall-mean-score: 0.747
f1-mean-score: 0.716

"""




















