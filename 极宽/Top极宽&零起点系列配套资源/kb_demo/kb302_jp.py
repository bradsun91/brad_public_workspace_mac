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
fjpy='data/USDJPY2018.csv'
feur='data/USDEUR2018.csv'
print('\n1# f,',fjpy,feur)
jpy=pd.read_csv(fjpy,index_col=0)
eur=pd.read_csv(feur,index_col=0)
zt.prDF('#1.1 @jpy',jpy)
#
zt.prDF('#1.2 @eur',eur)
#--

#2
#2.1
print('\n2# train')
df2=pd.DataFrame()

df2['jpy']=jpy['Close']
df2['eur']=eur['Close']
zt.prDF('#2.1 @df2',df2)
#
#2.2
df2['jpy']=df2['jpy']/100
zt.prDF('#2.2 @df2.x',df2)

#3
df3=df2.head(300)
df3=df3.sort_index()
zdr.drm_line(df3,'USDJPY v USDEUR 汇率对比曲线图',xlst=['jpy','eur'])


#-----------------------    
print('\nok!')
