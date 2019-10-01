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

from sklearn.neighbors import KNeighborsRegressor
from sklearn import ensemble 
#
import sklearn 



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
print('#1,rd data & init')
np.set_printoptions(suppress=True)
pd.set_option('display.width', 450)
    
#
clst=['open','high','low','close']

#
#fss='data/sh2018.csv'
fss='data/btc2018.csv'
xdf=pd.read_csv(fss,index_col=0)
xdf=xdf.sort_index()
zt.prDF('xdf',xdf)

#2
print('#2,xed data')
xdf['xprice']=xdf['close'].shift(-1)
#xdf['y']=xdf['xprice']
#   
xdf['kpr']=xdf['xprice']/xdf['close']*1000
#xdf['y']=0
xdf['close_next']=0
xdf.loc[xdf.kpr>1005,'close_next']=2
xdf.loc[xdf.kpr<995,'close_next']=1
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

        
#4
print('#4.1,准备AI数据')
x=df_train[clst].values
y=df_train['close_next'].values
#
xtst=df_test[clst].values


#

#5
print('#5,模型设置')
mx = ensemble.BaggingClassifier()
#6
print('#6,fit训练模型')
mx.fit(x,y)



#7
print('#7,predict模型预测数据')
df_test['close_pred']=mx.predict(xtst)
zt.prDF('df',df_test)

#8
print('#8,绘制对比数据图')
df_test[['close_next','close_pred']].plot(linewidth=3)
#print(predicted)

#9
print('#9,按1%精度验证模型')
dacc,df,xlst=zai.ai_acc_xed2ext(df_test['close_next'],df_test['close_pred'],1,True)

#10
print('\n#10,value_counts')
print("\ndf_test['close_next'].value_counts()")
print(df_test['close_next'].value_counts())
print("\ndf_test['close_pred'].value_counts()")
print(df_test['close_pred'].value_counts())
#

#-----------------------    
print('\nok!')

#-----------------------    
'''

54.26%,@mlst+clst
--
68.09%,@clst

@drx4,68.09%;
@drx3,67.02%; 
@drx2,acc: 63.83%;
'''