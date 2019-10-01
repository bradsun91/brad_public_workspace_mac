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
import ffn

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
print('#1,rd data')
rss='data/'
stklst=['000020','000528','000978','300020','300035','600201','600211']
stkPools=zdat.pools_frd4lst(rss,stklst)
#print(stkPools)

#2
print('#2,edit data')
df9=zdat.pools_lnk4lst(stkPools,stklst,'close')
zt.prDF('df9',df9)

#3
print('#3,cut data time')
tim0str,tim9str='2010-01-01','2015-12-31'
df2=zdat.df_kcut8tim(df9,'',tim0str,tim9str)
df2=df2.sort_index()
zt.prDF('df2',df2)    
df2.plot()

#4
print('#4,rebase')

df3=df2.rebase()
zt.prDF('df3',df3)    
df3.plot()

#-----------------------    
print('\nok!')
