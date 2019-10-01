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

#-----------------------

#1 
fss='data/iris.csv'
df=pd.read_csv(fss,index_col=False)
#2
df.loc[df['xname']=='virginica', 'xid'] = 1
df.loc[df['xname']=='setosa', 'xid'] = 2
df.loc[df['xname']=='versicolor', 'xid'] = 3
df['xid']=df['xid'].astype(int)
df.to_csv('tmp/iris2.csv',index=False)

#3
print('\n3#df')       
print(df.tail())
print(df.describe())

#4
d10=df['xname'].value_counts()
print('\n4#xname')       
print(d10)       

#5
d10=df['xid'].value_counts()
print('\n5#xid')       
print(d10)       

        
#-----------------------    
print('\nok!')
