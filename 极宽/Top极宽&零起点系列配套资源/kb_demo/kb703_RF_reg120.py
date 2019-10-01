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
import numpy as np
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
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import tree
from sklearn import ensemble

from sklearn.externals.six import StringIO
import pydotplus as pydot
#from IPython.display import Image  
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

    
        

#1 
print('#1,rd data')
fss='data/btc2018.csv'
xdf=pd.read_csv(fss,index_col=0)
xdf=xdf.sort_index()
zt.prDF('xdf',xdf)

#2
print('#2,xed data')
xdf['close_next']=xdf['close'].shift(-1)
xdf.fillna(method='pad',inplace=True)
zt.prDF('xdf',xdf)

#3.1
print('#3,cut data')
tim0str,tim9str='2010-10-10','2017-12-31'
df_train=zdat.df_kcut8tim(xdf,'',tim0str,tim9str)
zt.prDF('#3.1 df_train\n',df_train)
#
#3.2
df_test=xdf[xdf.index>'2018'] #.tail(100)
zt.prDF('#3.2 df_test\n',df_test)

#4,
print('#4,模型测试')
mfun=ensemble.RandomForestRegressor
#zai.mxtst_DTree010(mfun,df_train,df_test,ntree=100,ndepth=3,ftg='tmp/pic001.png')

df9=zai.mxtst_DTree100(mfun,df_train,df_test,ndepth=3)
df9['k3']=df9['k']
df2=zai.mxtst_DTree100(mfun,df_train,df_test,ndepth=5)
df9['k5']=df2['k']
df2=zai.mxtst_DTree100(mfun,df_train,df_test,ndepth=10)
df9['k10']=df2['k']
zt.prDF('df9',df9)
df9[['k3','k5','k10']].plot(linewidth=3)

#-----------------------    
print('\nok!')


'''

 df
           k  ntree  ndepth
ntree                      
50.0   21.28   50.0     3.0
100.0  22.34  100.0     3.0
150.0  12.77  150.0     3.0
200.0  26.60  200.0     3.0
250.0  25.53  250.0     3.0
300.0  21.28  300.0     3.0
350.0  23.40  350.0     3.0
400.0  22.34  400.0     3.0
450.0  24.47  450.0     3.0
500.0  24.47  500.0     3.0

len-DF: 10

----------------
           k  ntree  ndepth
ntree                      
50.0   19.15   50.0     5.0
100.0  22.34  100.0     5.0
150.0  27.66  150.0     5.0
200.0  12.77  200.0     5.0
250.0  22.34  250.0     5.0
300.0  26.60  300.0     5.0
350.0  24.47  350.0     5.0
400.0  25.53  400.0     5.0
450.0  24.47  450.0     5.0
500.0  24.47  500.0     5.0

len-DF: 10
----------------

 df
           k  ntree  ndepth
ntree                      
50.0   19.15   50.0    10.0
100.0  25.53  100.0    10.0
150.0  22.34  150.0    10.0
200.0  14.89  200.0    10.0
250.0  25.53  250.0    10.0
300.0  21.28  300.0    10.0
350.0  25.53  350.0    10.0
400.0  24.47  400.0    10.0
450.0  23.40  450.0    10.0
500.0  26.60  500.0    10.0

len-DF: 10
'''
