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
#xdf['close_next']=xdf['close'].shift(-1)
xdf['xprice']=xdf['close'].shift(-1)
#xdf['y']=xdf['xprice']
#   
xdf['kpr']=xdf['xprice']/xdf['close']*1000
xdf['close_next']=0
xdf.loc[xdf.kpr>1005,'close_next']=2
xdf.loc[xdf.kpr<995,'close_next']=1
xdf.fillna(method='pad',inplace=True)
zt.prDF('xdf',xdf)
#

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
mfun=ensemble.ExtraTreesClassifier
zai.mxtst_DTree010(mfun,df_train,df_test,ntree=100,ndepth=3,ftg='tmp/pic001.png')


#-----------------------    
print('\nok!')

