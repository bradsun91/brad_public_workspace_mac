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
import numpy as np
import pandas as pd
import plotly  as py
import plotly.graph_objs as go 
import plotly.figure_factory  as pyff

#
import sklearn 
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn import cross_validation

#
import zsys
import ztools as zt
import ztools_str as zstr
import ztools_web as zweb
import ztools_data as zdat
import ztools_draw as zdr
import ztools_ai as zai
import zpd_talib as zta

#
#-----------------------

#-----------------------

#1 
print('#1,rd data')
fss='data/sh2018.csv'
xdf=pd.read_csv(fss,index_col=0)
xdf=xdf.sort_index()
zt.prDF('xdf',xdf)

#2
print('#2,xed data')
xdf['y']=xdf['close'].shift(-1)
zt.prDF('xdf#2.1',xdf)
xdf.fillna(method='pad',inplace=True)
zt.prDF('xdf#2.2',xdf)

        
#3
print('#3,cut data')
df=xdf[xdf.index>'2018'] 


#4
print('#4,准备AI数据')
clst=['open','high','low','close']
x=df[clst].values
y=df['y'].values

#5
print('#5,模型设置')
mx = linear_model.LinearRegression()


#6
print('#6,交叉验证')
c10 = cross_validation.cross_val_score(mx,x,y, cv=2)
print('cross_val_score：',c10)
print('mean：',np.mean(c10))

#-----------------------    
print('\nok!')

