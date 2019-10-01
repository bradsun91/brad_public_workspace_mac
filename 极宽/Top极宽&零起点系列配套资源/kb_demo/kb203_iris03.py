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

import pandas as pd

#
import sklearn 
from sklearn.cross_validation import train_test_split
#
#-----------------------

#1 
fss='data/iris2.csv'
df=pd.read_csv(fss,index_col=False)

#2
print('\n2# df')       
print(df.tail())


#3
xlst,ysgn=['x1','x2','x3','x4'],'xid'
x,y= df[xlst],df[ysgn]  
#
print('\n3# xlst,',xlst)
print('ysgn,',ysgn)
print('x')
print(x.tail())
print('y')
print(y.tail())

#4
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
x_test.index.name,y_test.index.name='xid','xid'
print('\n4# type')
print('type(x_train),',type(x_train))
print('type(x_test),',type(x_test))
print('type(y_train),',type(y_train))
print('type(y_test),',type(y_test))

#5
fs0='tmp/iris_'
print('\n5# fs0,',fs0)
x_train.to_csv(fs0+'xtrain.csv',index=False);
x_test.to_csv(fs0+'xtest.csv',index=False)
y_train.to_csv(fs0+'ytrain.csv',index=False,header=True)
y_test.to_csv(fs0+'ytest.csv',index=False,header=True)

#6
print('\n6# x_train')
print(x_train.tail())
print('\nx_test')
print(x_test.tail())

#7
print('\n7# y_train')
print(y_train.tail())
print('\ny_test')
print(y_test.tail())

#-----------------------    
print('\nok!')
 