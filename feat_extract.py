#coding=utf-8
import pandas as pd
import numpy as np
import time

csv1 = file("/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/train.csv", 'rb')
csv2 = file("/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/all_user_yongdian_data_2015.csv",'rb')
df1=pd.read_csv(csv1,header=None,names=["uid","label"])
df2=pd.read_csv(csv2)

group2 = df2.groupby("CONS_NO",as_index='false')
KWH=group2["KWH"].agg([np.sum,np.mean,np.max,np.min,np.std])
# print KWH_avg.index.names
KWH.insert(0,'id_feature',KWH.index%10000)
print KWH.keys()

# df2=df2.pivot_table(index="CONS_NO",values="KWH",columns=["DATA_DATE"])
# df2.insert(0,'KWH_avg',KWH_avg.values)
# df2.insert(0,'id_feature',df2.index%10000)


sampler = np.random.permutation(len(df1))
# df1=df1.take(sampler)

# train=pd.merge(df1,df2,right_index=True,left_on='uid')
train=pd.merge(df1,KWH,right_index=True,left_on='uid')

train=train.fillna(0)
train.to_csv('/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/traindata3.csv',header=False,index=False)

# csv3=file("/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/test.csv", 'rb')
# df3=pd.read_csv(csv3,header=None,names=["uid"])
# predict=pd.merge(df3,KWH,right_index=True,left_on='uid')
# predict=predict.fillna(0)
# predict.to_csv('/Users/zhangdenghui/Documents/CCF_contest/BehaviorAnalysis/predictdata2.csv',header=False,index=False)