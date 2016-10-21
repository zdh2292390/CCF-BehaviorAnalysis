import xgboost as xgb
import pandas as pd
import numpy as np
import csv
import time

csv1 = file("/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/traindata2.csv", 'rb')
csv2 = file("/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/predictdata2.csv", 'rb')
traindata = np.loadtxt(csv1,delimiter=",")
# x_train= traindata[:, 2:]
# y_train = traindata[:, 1]

x_train= traindata[:5000, 2:]
y_train = traindata[:5000, 1]

x_test = traindata[5001:,2:]
y_test = traindata[5001:, 1]

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
param = {'bst:max_depth':3, 'bst:eta':0.1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = "map"
evallist  = [(dtest,'eval')]
num_round = 40
eval_re={}
bst = xgb.train( param, dtrain, num_round, evals=evallist ,evals_result=eval_re)


predictdata = np.loadtxt(csv2,delimiter=",")
x_predict= predictdata[:, 1:]
dpre = xgb.DMatrix(x_predict)
ypred = bst.predict(dpre)
print type(ypred)
result=[]
for i in range(len(ypred)):
    result.append([int(predictdata[i][0]),ypred[i]])

result.sort(key=lambda x:x[1],reverse=True)
result_nolabel=[]
for i in range(len(result)):
    result_nolabel.append([int(result[i][0])])

resultfile = file('/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/predict_'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))+'.csv', 'wb')
writer = csv.writer(resultfile)
writer.writerows(result_nolabel)
resultfile.close()