#coding=utf-8
'''
TopQuant-简称TQ极宽智能量化回溯分析系统，培训课件-配套教学python课件程序

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2017.10.1 首发

网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713

'''

import sys;
sys.path.append("topqt/")
#
import pandas as pd

#
import sklearn 
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

#
import zsys
import ztools as zt
import ztools_str as zstr
import ztools_web as zweb
import ztools_data as zdat
import ztools_ai as zai
import zpd_talib as zta

#
#-----------------------

#1 
fs0='data/iris_'
print('\n1# fs0,',fs0)
x_train=pd.read_csv(fs0+'xtrain.csv',index_col=False);
y_train=pd.read_csv(fs0+'ytrain.csv',index_col=False);


#2
print('\n2# train')
print(x_train.tail())
print(y_train.tail())


#3
print('\n3# 建模')
mx =zai.mx_line(x_train.values,y_train.values)

#4 
x_test=pd.read_csv(fs0+'xtest.csv',index_col=False)
df9=x_test.copy()
print('\n4# x_test')
print(x_test.tail())

#5
print('\n5# 预测')
y_pred = mx.predict(x_test.values)
df9['y_predsr']=y_pred

#6
y_test=pd.read_csv(fs0+'ytest.csv',index_col=False)
print('\n6# y_test')
print(y_test.tail())


#7
df9['y_test'],df9['y_pred']=y_test,y_pred
df9['y_pred']=round(df9['y_predsr']).astype(int)   
df9.to_csv('tmp/iris_9.csv',index=False)
print('\n7# df9')
print(df9.tail())

   
#
#8   
dacc,df9x=zai.ai_acc_xed2x(df9['y_test'],df9['y_pred'],1,False)
print('\n8.1# mx:mx_sum,kok:{0:.2f}%'.format(dacc))   
#
print('\n8.2# df9x')
print(df9x.tail())

#-----------------------    
print('\nok!')
