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

#-----------------------

#1 
print('#1,rd data')
fss='data/sh2018.csv'
xdf=pd.read_csv(fss,index_col=0)
xdf=xdf.sort_index()
zt.prDF('xdf',xdf)

#2
print('#2,xed data')
xdf['xprice']=xdf['close'].shift(-1)
#xdf['y']=xdf['xprice']
#   
xdf['kpr']=xdf['xprice']/xdf['close']*1000
xdf['y']=0
xdf.loc[xdf.kpr>1005,'y']=2
xdf.loc[xdf.kpr<995,'y']=1
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
print('#4,准备AI数据')

clst=['open','high','low','close']
x=df_train[clst].values
y=df_train['y'].values
#
xtst=df_test[clst].values

#5
print('#5,模型设置')
#mx = tree.DecisionTreeRegressor(max_depth=50)
mx = ensemble.GradientBoostingClassifier(n_estimators=100,max_depth=3)

#6
print('#6,fit训练模型')
mx.fit(x,y)
'''
#7
print('#7,输出模型')
vlst=['up','down','eq']
dot_data = StringIO()
tree.export_graphviz(mx, out_file=dot_data,  
            feature_names=clst,  
            class_names=vlst,  
            filled=True, rounded=True,  
            special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
#graph.write_pdf('tmp/tree01.pdf')
graph.write_png('tmp/tree02.png')
'''
#8
print('#8,predict模型预测数据')
df_test['y2']=mx.predict(xtst) #cross_val_predict(mx,x, y, cv=20)
zt.prDF('df',df_test)

#9
print('#9,验证模型预测效果')
#9.1
print('\n#9.1,按5%精度验证模型')
dacc,df=zai.ai_acc_xed2x(df_test['y'],df_test['y2'],5,True)
print('acc',dacc)

#9.2
print('\n#9.2,按1%精度验证模型')
dacc,df,xlst=zai.ai_acc_xed2ext(df_test['y'],df_test['y2'],1,True)

#9.3
print('\n#9.3,value_counts')
print("\ndf_train['y'].value_counts()")
print(df_train['y'].value_counts())
#
print("\ndf_test['y'].value_counts()")
print(df_test['y'].value_counts())
print("\ndf_test['y2'].value_counts()")
print(df_test['y2'].value_counts())

#-----------------------    
print('\nok!')

#-----------------------    

'''
n_estimators
-------------@@n=10

#9.1,按5%精度验证模型
acc 40.426

#9.2,按1%精度验证模型
ky0=1; n_df9,94,n_dfk,38
acc: 40.43%;  MSE:1.52, MAE:0.90,  RMSE:1.23, r2score:-1.18, @ky0:1.00

-------------@@n=50
#9.1,按5%精度验证模型
acc 40.426

#9.2,按1%精度验证模型
ky0=1; n_df9,94,n_dfk,38
acc: 40.43%;  MSE:1.52, MAE:0.90,  RMSE:1.23, r2score:-1.18, @ky0:1.00
-------------@@n=100

#9.1,按5%精度验证模型
acc 42.553

#9.2,按1%精度验证模型
ky0=1; n_df9,94,n_dfk,40
acc: 42.55%;  MSE:1.47, MAE:0.87,  RMSE:1.21, r2score:-1.10, @ky0:1.00

-------------@@n=300
#9.1,按5%精度验证模型
acc 44.681

#9.2,按1%精度验证模型
ky0=1; n_df9,94,n_dfk,42
acc: 44.68%;  MSE:1.29, MAE:0.80,  RMSE:1.13, r2score:-0.84, @ky0:1.00

-------------@@n=500
#9.1,按5%精度验证模型
acc 41.489

#9.2,按1%精度验证模型
ky0=1; n_df9,94,n_dfk,39
acc: 41.49%;  MSE:1.41, MAE:0.86,  RMSE:1.19, r2score:-1.02, @ky0:1.00

'''

