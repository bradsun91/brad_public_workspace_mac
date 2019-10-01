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
import ffn
import pandas as pd
import plotly  as py
import plotly.graph_objs as go 
import plotly.figure_factory  as pyff

#
import sklearn 
from sklearn import preprocessing
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
import ztools_draw as zdr
import ztools_ai as zai
import zpd_talib as zta

#
#-----------------------

#-----------------------

#1 
print('#1,rd data')
rss='data/'
stklst=['btc2018','eth2018','doge2018']
stkPools=zdat.pools_frd4lst(rss,stklst)

#2
print('#2,edit data')
df9=zdat.pools_lnk4lst(stkPools,stklst,'close')
zt.prDF('df9',df9)

#3
print('#3,cut data time')
tim0str,tim9str='2016-01-01','2016-12-31'
df2=zdat.df_kcut8tim(df9,'',tim0str,tim9str)
df2=df2.sort_index()
#
df2=df2.tail(10)
zt.prDF('df2',df2)    
df2.plot()

#4

print('#4,edit data')
x=df2[stklst].values
print('x',x)
xfun = preprocessing.MinMaxScaler()
x10= xfun.fit_transform(x)
print('x10',x10)
#5
print('#5,data anz')
x_mean=np.round(x10.mean(axis=0),2)
x_std=x10.std(axis=0)
print('x_mean',x_mean)
print('x_std',x_std)


#6
print('#6,data anz')
x20=x10.T
print('x20',x20)
#
df3=pd.DataFrame()
df3['btc'],df3['eth'],df3['dog']=x20[0],x20[1],x20[2]
zt.prDF('df3',df3)
df3.plot()



#-----------------------    
print('\nok!')

