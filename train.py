#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from  sklearn.metrics import average_precision_score
import time
import csv

csv1 = file("/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/traindata.csv", 'rb')
# traindata=pd.read_csv(csv1)
traindata = np.loadtxt(csv1,delimiter=",")
# x_train= traindata[:8000, 2:]
# y_train = traindata[:8000, 1]
x_train= traindata[:, 2:]
y_train = traindata[:, 1]

x_test = traindata[8000:,2:]
y_test = traindata[8000:, 1]
print len(x_train)
print len(x_test)
# print x_train.head(3)
clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,max_depth=3, random_state=0).fit(x_train, y_train)

print clf.score(x_test, y_test)
print clf.classes_
csv2 = file("/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/predictdata.csv", 'rb')
predictdata = np.loadtxt(csv2,delimiter=",")
x_predict= predictdata[:, 1:]
# y_predict=clf.predict(x_predict)
y_predict=clf.predict_proba(x_predict)
result=[]
for i in range(len(y_predict)):
    result.append([int(predictdata[i][0]),y_predict[i][1]])

result.sort(key=lambda x:x[1],reverse=True)


result_nolabel=[]
for i in range(len(result)):
    result_nolabel.append([int(result[i][0])])

resultfile = file('/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/predict_'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))+'.csv', 'wb')
writer = csv.writer(resultfile)
# writer.writerow(['uid', 'label'])
writer.writerows(result_nolabel)
resultfile.close()

