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
import ztools_draw as zdr
import ztools_ai as zai

import zpd_talib as zta

#
#-----------------------

#1 
fbtc='data/btc2018.csv'
fdog='data/doge2018.csv'
print('\n1# f,',fbtc,fdog)
btc=pd.read_csv(fbtc,index_col=0)
dog=pd.read_csv(fdog,index_col=0)
zt.prDF('#1.1 @btc',btc)
#
zt.prDF('#1.2 @dog',dog,nfloat=3)
#--
#2
print('\n2# train')
df2=pd.DataFrame()      
df2['btc']=btc['close']
df2['dog']=dog['close']
df2['btc1k']=df2['btc']/1000
df2['dog1k']=df2['dog']*1000
zt.prDF('#2 @df2.x',df2,nfloat=3)

#3
df3=df2.head(300)
df3=df3.sort_index()
zdr.drm_line(df3,'BTC v DOG价格曲线图',xlst=['btc1k','dog1k'])


#-----------------------    
print('\nok!')
