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
import plotly  as py
import plotly.graph_objs as go 
import plotly.figure_factory  as pyff

#
import sklearn 
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
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
#4
print('#4,准备AI数据')

clst=['open','high','low','close']
x0=df_train[clst].values
y=df_train['y'].values

#
xtst0=df_test[clst].values

#5
print('#5,模型设置')
mx = linear_model.LinearRegression()


# 5.b
print('#5,多项式参数变换')
poly = PolynomialFeatures(degree=2,include_bias=False)
x = poly.fit_transform(x0)
xtst = poly.fit_transform(xtst0)

#6
print('#6,fit训练模型')
mx.fit(x,y)

#7
print('#7,predict模型预测数据')
      
df_test['y2']=mx.predict(xtst) #cross_val_predict(mx,x, y, cv=20)
#predicted = cross_val_predict(mx,x, y, cv=10)
#df['y2']=predicted
zt.prDF('df',df_test)
df_test[['y','y2']].plot()
#print(predicted)

#8.1
print('#8,验证模型预测效果')
print('\n#8.1,按5%精度验证模型')
dacc,df=zai.ai_acc_xed2x(df_test['y'],df_test['y2'],5,True)
print('acc',dacc)

#8.2
print('\n#8.2,按1%精度验证模型')
dacc,df,xlst=zai.ai_acc_xed2ext(df_test['y'],df_test['y2'],1,True)


#-----------------------    
print('\nok!')

