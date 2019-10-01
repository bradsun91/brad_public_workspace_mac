# -*- coding: utf-8 -*- 
'''
TopQuant-简称TQ极宽智能量化回溯分析系统，培训课件-配套教学python程序

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2017.10.1 首发


网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  
  
文件名:ztools_msta.py
默认缩写：import ztools_msta as zmsta
简介：Top极宽量化·回溯策略模块（多策略）
 

'''
#

import sys,os,re
import arrow,bs4,random
import numexpr as ne  
import numpy as np
import pandas as pd
import tushare as ts
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
import sklearn
from sklearn import metrics
#
#import tflearn
#import tensorflow as tf

#
import zsys
import zpd_talib as zta
import ztools as zt
import ztools_tq as ztq
import ztools_str as zstr
import ztools_data as zdat
import ztools_sta as zsta




#-------------------

#----------mx_misc
def mx_sta4list(qx,mx):
    fsta0=mx.pop(0)
    mx=list(map(lambda x:int(x), mx))
        
    #[staNam,v,v2,v3,v4]=mx
    fsta='zsta.%s'%fsta0
    #'ccxt.%s ()' % xid)
    #mx=list(map(lambda x:int(x), mx))
    #print(fsta,'@m',mx)
    #xx
    #eval ('ccxt.%s ()' % xid)
    qx.staFun,qx.staVars=eval(fsta),mx
    #

  
#----------mx_sta
    


def mx_sta010(qx):
    '''
    qx.staVars=[0,0,25,9]    
    '''
    #
    qx.trd_MulStaFlag=True
    df0,ksgn=qx.wrkSybDat,qx.priceSgn
    vsta=qx.staVars
    [v,v2,v3,v4]=vsta
    #
    #mx100=[ ['Low10dp',2,0,100,105], ['UpDown',4,4,100,100],
    #     ['Low10dp',10,0,100,105], ['UpDown',30,30,104,101] ]
    #           
    #mx100=zdat.df2list(qx.mstaPools)
    mxnum=len(qx.mstaPools)
    #print('p',qx.mstaPools)
    #@bt,qx.staFun(qx)
    #
    #qx0.mstaPools=mx10
    mlst=[]
    for xc in range(mxnum):
        mx=qx.mstaPools[xc].copy()
        msgn='ktrd{0:03}'.format(xc)
        mlst.append(msgn)
        #
        mx_sta4list(qx,mx)
        #
        #qx.staFun,qx.staVars=fsta,mx
        #print(xc,'/',mxnum,mx,msgn,fsta)
        #qx.staFun,qx.staVars=eval (mx[0]),mx[1]
        qx.wrkSybDat=df0
        qx.usr_num9,qx.usrBuyMoney9,qx.usrMoney=0,0,qx.usrMoney0
        #
        qx.staFun(qx)
        #
        df0[msgn]=qx.wrkSybDat['ktrd']
        #
    #    
    df=df0
    df['mtrd9']=df[mlst].sum( axis=1)
    df['kmtrd']=round(df['mtrd9']/mxnum*100,2)
    
    #zt.prDF('df0',df,5);print(df.describe())
    #print('@mx',mx100,v3,v4,mxnum)
    #xxx
    #
    qx.trd_MulStaFlag=False
    #
    #df['mtrd']=0
    #df['mtrd']=0  # =1:buy; =-1:sell
    #df.loc[(df.kmtrd>v3)&(df.kmtrd),'mtrd']=1    #buy
    #df.loc[df.kmax<-v4,'ktrd']=-1   #sell
    #
    zsta.trd_sub(qx,df,['kmtrd','>',v3, 'kmtrd','<',-v4])
    #
    #qx.staVars=vsta
    
    #

    #
    return qx        



def mx_sta010ss(qx):
    '''
    qx.staVars=[0,0,25,9]    
    '''
    #
    qx.trd_MulStaFlag=True
    df0,ksgn=qx.wrkSybDat,qx.priceSgn
    vsta=qx.staVars
    [v,v2,v3,v4]=vsta
    #
    #mx100=[ ['Low10dp',2,0,100,105], ['UpDown',4,4,100,100],
    #     ['Low10dp',10,0,100,105], ['UpDown',30,30,104,101] ]
    #           
    mx100=zdat.df2list(qx.mstaPools)
    mxnum=len(mx100)
    #@bt,qx.staFun(qx)
    #
    #qx0.mstaPools=mx10
    mlst=[]
    for xc, mx in enumerate(mx100):
        msgn='ktrd{0:03}'.format(xc)
        mlst.append(msgn)
        #
        fsta=mx.pop(0)
        mx=list(map(lambda x:int(x), mx))
        #print('v',v,v2,v3,v4)
        qx.staFun,qx.staVars=eval(fsta),mx
        #print(xc,'/',mxnum,mx,msgn,fsta)
        #qx.staFun,qx.staVars=eval (mx[0]),mx[1]
        qx.wrkSybDat=df0
        qx.usr_num9,qx.usrBuyMoney9,qx.usrMoney=0,0,qx.usrMoney0
        #
        qx.staFun(qx)
        #
        df0[msgn]=qx.wrkSybDat['ktrd']
        #
    #    
    df=df0
    df['mtrd9']=df[mlst].sum( axis=1)
    df['kmtrd']=round(df['mtrd9']/mxnum*100,2)
    
    #zt.prDF('df0',df,5);print(df.describe())
    #print('@mx',mx100,v3,v4,mxnum)
    #xxx
    #
    qx.trd_MulStaFlag=False
    #
    #df['mtrd']=0
    #df['mtrd']=0  # =1:buy; =-1:sell
    #df.loc[(df.kmtrd>v3)&(df.kmtrd),'mtrd']=1    #buy
    #df.loc[df.kmax<-v4,'ktrd']=-1   #sell
    #
    zsta.trd_sub(qx,df,['kmtrd','>',v3, 'kmtrd','<',-v4])
    #
    #qx.staVars=vsta
    
    #

    #
    return qx        
    
    