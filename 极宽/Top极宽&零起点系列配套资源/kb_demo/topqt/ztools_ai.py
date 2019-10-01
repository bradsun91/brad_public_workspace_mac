# -*- coding: utf-8 -*- 
'''
TopQuant-简称TQ极宽智能量化回溯分析系统，培训课件-配套教学python程序

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2017.10.1 首发


网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  
  
文件名:ztools_ai.py
默认缩写：import ztools_ai as zai
简介：Top极宽量化·常用AI工具函数集
 

'''
#

import sys,os,re,pickle
import arrow,bs4,random,copy
import numexpr as ne  
import numpy as np
import pandas as pd
import tushare as ts
import pandas_datareader.data as pdat     # 网络财经数据接口库

#import talib as ta

import pypinyin 
#

import matplotlib as mpl
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
#import multiprocessing
#
'''
import keras as ks

import sklearn
from sklearn import metrics
#
import keras
from keras.models import Sequential,load_model
from keras.utils import plot_model
#
import tflearn
#import tensorflow as tf
'''
#
import zsys
import ztools as zt
import ztools_str as zstr
import ztools_data as zdat




#-------------------
#
import zpd_talib as zta

#
#-------------------


#---------------------------ai.xxx
def ai_varRd(fmx0):
    fvar=fmx0+'tqvar.pkl'
    qx=zt.f_varRd(fvar)
    for xkey in qx.aiMKeys:
        fss=fmx0+xkey+'.mx'
        mx=load_model(fss)
        qx.aiModel[xkey]=mx
    #
    return qx
    
def ai_varWr(qx,fmx0):    
    fvar=fmx0+'tqvar.pkl'
    mx9=qx.aiModel
    qx.aiMKeys=list(mx9.keys())
    qx.aiModel={}
    zt.f_varWr(fvar,qx)
    print('fvar,',fvar)
    #
    for xkey in mx9:
        fss=fmx0+xkey+'.mx'
        mx9[xkey].save(fss)
        print('fmx,',fss)
    #
    qx.aiModel=mx9
    
#---------------------------ai.xxx
    
#---------------------------ai.dacc
def ai_acc_xed2x(y_true,y_pred,ky0=5,fgDebug=False):
    '''
    效果评估函数，用于评估机器学习算法函数的效果。
    输入：
    	y_true,y_pred，pandas的Series数据列格式。
    	ky0，结果数据误差k值，默认是5，表示百分之五。
    	fgDebug，调试模式变量，默认为False。
    返回：
        dacc,准确率，float格式
        df，结果数据，pandas列表格式DataFrame
    
    '''
    #1
    df,dacc=pd.DataFrame(),-1
    #print('n,',len(y_true),len(y_pred))
    if (len(y_true)==0) or (len(y_pred)==0):
        #print('n,',len(y_true),len(y_pred))
        return dacc,df
        
    #
    y_num=len(y_true)
    #df['y_true'],df['y_pred']=zdat.ds4x(y_true,df.index),zdat.ds4x(y_pred,df.index)
    df['y_true'],df['y_pred']=pd.Series(y_true),pd.Series(y_pred)
    df['y_diff']=np.abs(df.y_true-df.y_pred)
    #2
    df['y_true2']=df['y_true']
    df.loc[df['y_true'] == 0, 'y_true2'] =0.00001
    df['y_kdif']=df.y_diff/df.y_true2*100
    #3
    dfk=df[df.y_kdif<ky0]   
    knum=len(dfk['y_pred'])
    dacc=knum/y_num*100
    #
    #5
    dacc=round(dacc,3)
    return dacc,df

def ai_acc_xed2ext(y_true,y_pred,ky0=5,fgDebug=False):
    '''
    效果评估函数，用于评估机器学习算法函数的效果。
    输入：
    	y_true,y_pred，pandas的Series数据列格式。
    	ky0，结果数据误差k值，默认是5，表示百分之五。
    	fgDebug，调试模式变量，默认为False。
    返回：
        dacc,准确率，float格式
        df，结果数据，pandas列表格式DataFrame
        [dmae,dmse,drmse,dr2sc]，各种扩充评估数据
    
    '''    
    #1
    df,dacc=pd.DataFrame(),-1
    if (len(y_true)==0) or (len(y_pred)==0):
        #print('n,',len(y_true),len(y_pred))
        return dacc,df
        
    #2
    y_num=len(y_true)
    #df['y_true'],df['y_pred']=zdat.ds4x(y_true,df.index),zdat.ds4x(y_pred,df.index)
    df['y_true'],df['y_pred']=y_true,y_pred
    df['y_diff']=np.abs(df.y_true-df.y_pred)
    #3
    df['y_true2']=df['y_true']
    df.loc[df['y_true'] == 0, 'y_true2'] =0.00001
    df['y_kdif']=df.y_diff/df.y_true2*100
    #4
    dfk=df[df.y_kdif<ky0]   
    knum=len(dfk['y_pred'])
    dacc=knum/y_num*100
    #
    #5
    dmae=metrics.mean_absolute_error(y_true, y_pred)
    dmse=metrics.mean_squared_error(y_true, y_pred)
    drmse=np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    dr2sc=metrics.r2_score(y_true,y_pred)
    #    
    #6
    if fgDebug:
        #print('\nai_acc_xed')
        #print(df.head())
        #y_test,y_pred=df['y_test'],df['y_pred']
        print('ky0={0}; n_df9,{1},n_dfk,{2}'.format(ky0,y_num,knum))
        print('acc: {0:.2f}%;  MSE:{1:.2f}, MAE:{2:.2f},  RMSE:{3:.2f}, r2score:{4:.2f}, @ky0:{5:.2f}'.format(dacc,dmse,dmae,drmse,dr2sc,ky0))
        
    #
    #7
    dacc=round(dacc,3)
    xlst=[dmae,dmse,drmse,dr2sc]
    return dacc,df,xlst

#---------------------------ai.model.xxx    
def ai_mul_var_tst(mx,df_train,df_test,nepochs=200,nsize=128,ky0=5):
    x_train,y_train=df_train['x'].values,df_train['y'].values
    x_test, y_test = df_test['x'].values,df_test['y'].values
    #
    mx.fit(x_train, y_train, epochs=nepochs, batch_size=nsize)
    #
    y_pred = mx.predict(x_test)
    df_test['y_pred']=zdat.ds4x(y_pred,df_test.index,True)
    dacc,_=ai_acc_xed2x(df_test.y,df_test['y_pred'],ky0,False)
    #
    return dacc
    

def ai_mx_tst_epochs(f_mx,f_tg,df_train,df_test,kepochs=100,nsize=128,ky0=5):
    ds,df={},pd.DataFrame()
    for xc in range(1,11):
        print('\n#',xc)
        dnum=xc*kepochs
        mx=ks.models.load_model(f_mx)
        t0=arrow.now()
        dacc=ai_mul_var_tst(mx,df_train,df_test,dnum,nsize,ky0=ky0)
        tn=zt.timNSec('',t0)
        ds['nepoch'],ds['epoch_acc'],ds['ntim']=dnum,dacc,tn
        df=df.append(ds,ignore_index=True)    
        
    #
    df=df.dropna()
    df['nepoch']=df['nepoch'].astype(int)
    print('\ndf')
    print(df)
    print('\nf,',f_tg)
    df.to_csv(f_tg,index=False)
    #
    df.plot(kind='bar',x='nepoch',y='epoch_acc',rot=0)
    df.plot(kind='bar',x='nepoch',y='ntim',rot=0)
    #
    return df



def ai_mx_tst_bsize(f_mx,f_tg,df_train,df_test,nepochs=500,ksize=32,ky0=5):
    ds,df={},pd.DataFrame()
    for xc in range(1,11):
        print('\n#',xc)
        dnum=xc*ksize
        mx=ks.models.load_model(f_mx)
        t0=arrow.now()
        dacc=ai_mul_var_tst(mx,df_train,df_test,nepochs,dnum,ky0=ky0)
        tn=zt.timNSec('',t0)
        ds['bsize'],ds['size_acc'],ds['ntim']=dnum,dacc,tn
        df=df.append(ds,ignore_index=True)    
        
    #
    df=df.dropna()
    df['bsize']=df['bsize'].astype(int)
    print('\ndf')
    print(df)
    print('\nf,',f_tg)
    df.to_csv(f_tg,index=False)
    #
    df.plot(kind='bar',x='bsize',y='size_acc',rot=0)
    df.plot(kind='bar',x='bsize',y='ntim',rot=0)
    return df

    
def ai_mx_tst_kacc(f_mx,f_tg,df_train,df_test,nepochs=500,nsize=128):
    ds,df={},pd.DataFrame()
    for xc in range(1,11):
        print('\n#',xc)
        dnum=xc*1
        mx=ks.models.load_model(f_mx)
        dacc=ai_mul_var_tst(mx,df_train,df_test,nepochs,nsize,ky0=dnum)
        ds['kacc'],ds['dacc']=dnum,dacc
        df=df.append(ds,ignore_index=True)    
        
    #
    df=df.dropna()
    df['kacc']=df['kacc'].astype(int)
    print('\ndf')
    print(df)
    print('\nf,',f_tg)
    df.to_csv(f_tg,index=False)
    #
    df.plot(kind='bar',x='kacc',y='dacc',rot=0)
    #
    return df

    

    