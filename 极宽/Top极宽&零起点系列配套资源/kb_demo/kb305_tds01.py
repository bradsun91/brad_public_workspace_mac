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

        
#3.1
print('#3,cut data')
tim0str,tim9str='2010-10-10','2017-12-31'
df_train=zdat.df_kcut8tim(xdf,'',tim0str,tim9str)
zt.prDF('#3.1 df_train\n',df_train)
#
#3.2
df_test=xdf[xdf.index>'2018'] #.tail(100)
zt.prDF('#3.2 df_test\n',df_test)


#-----------------------    
print('\nok!')
