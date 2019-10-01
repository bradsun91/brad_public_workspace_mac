# -*- coding: utf-8 -*- 
'''
TopQuant-简称TQ极宽智能量化回溯分析系统，培训课件-配套教学python程序

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2017.10.1 首发


网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  
  
文件名:ztools_tq.py
默认缩写：import ztools_tq as ztq
简介：Top极宽量化·常用量化工具函数集
 

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




#-------init.TQ.xxx,qx.xxx
def tq_init(rs0,prjNam='TQ001',sybLst=[]):
    pd.set_option('display.width', 450)    
    pd.set_option('display.float_format', lambda x: '%.9g' % x)
    #pd.set_option('display.float_format', zt.xfloat9)    
    np.set_printoptions(suppress=True) #取消科学计数法 #as_num(1.2e-4)
    #
    qx=zsys.TQ_bar()
    qx.prjNam,qx.rdat0,qx.rdat=prjNam,rs0,rs0
    qx.rdatTyp,qx.priceSgn='5m','avg'
    qx.timDayFlag=(rs0.upper().find('MIN')<0)
    qx.timFmt=zt.iff2(qx.timDayFlag,'YYYY-MM-DD','YYYY-MM-DD HH:mm:ss')
    qx.timFmtInv=zt.iff2(qx.timDayFlag,'%Y-%m-%d','%Y-%m-%d %H:%M:%S')
    #
    qx.sybLst=sybLst
    qx.sybNum=len(sybLst)
    #
    print('tq_init name...')
    #
    return qx



def tq_prVar(qx):
    print('\nobj:qx')
    zt.prObj(qx)
    #
    print('\nzsys.xxx')
    print('    rdat0,',zsys.rdat0)
    print('    rdatCN,',zsys.rdatCN)
    print('    rdatCNX,',zsys.rdatCNX)
    print('    rdatMin0,',zsys.rdatMin0)
    print('    rdatTick,',zsys.rdatTick)
    #
    print('\ncode list:',qx.sybLst)
    #
    #zt.prx('btTimLst',qx.btTimLst)
    zt.prx('usrPools',qx.usrPools)      #用户股票池资产数据 字典格式
    print('\nusrMoney,usrTotal:',qx.usrMoney,qx.usrTotal)    
    #
    tq_prTrdlib(qx)
    #zt.prx('qx.trdLib',qx.trdLib.head())
    #zt.prx('qx.trdLib',qx.trdLib.tail())
    #


 
def tq_prWrk(qx):
    print('\n\t bt_main_1day,',qx.wrkSybCod,qx.wrkTimStr)
    #
    zt.prx('syb info',qx.wrkSybInfo)
    zt.prx('wrkSybDat.head',qx.wrkSybDat.head(10))
    zt.prx('wrkSybDat.tail',qx.wrkSybDat.tail(10))
    #
    zt.prx('btTimLst',qx.btTimLst)
    zt.prx('usrPools',qx.usrPools)      #用户股票池资产数据 字典格式
    print('\nusrMoney,usrTotal:',qx.usrMoney,qx.usrTotal)    
    #
    #zt.prx('qx.trdLib',qx.trdLib.head())
    #zt.prx('qx.trdLib',qx.trdLib.tail())
    
    
   

#-------tq.misc
def tq_kusdcny(kcny0=6.61):
    t0=arrow.now().shift(days=-10).format('YYYY-MM-DD HH:mm:ss')
    df = pdat.get_data_fred('DEXCHUS',t0)
    if len(df.index)>0:v=df['DEXCHUS'][-1]
    else:v=kcny0
    print('v',v)
    #
    return v

def tq_ntim4sgn(xsgn0):
    dnum,xsgn=0,xsgn0.lower()
    if xsgn=='1m':dnum=1440
    if xsgn=='5m':dnum=288
    if xsgn=='15m':dnum=96
    if xsgn=='30m':dnum=48
    if xsgn=='60m':dnum=24
    if xsgn=='1h':dnum=24
    #if xsgn=='2h':dnum=24
    if xsgn=='1d':dnum=1
    
    #
    return dnum

def tq_xtim2FRQ(xsgn0):
    '''
    常用的基础频率
    
    别名	偏移量	说明
    D/d	Day	每日历日
    B	BusinessDay	每工作日
    H/h	Hour	每小时
    T或min	Minute	每分
    S	Secend	每秒
    L或ms	Milli	每毫秒（每千分之一秒）
    U	Micro	每微秒（即百万分之一秒）
    M	MonthEnd	每月最后一个日历日
    BM	BusinessDayEnd	每月最后一个工作
    '''
    xsgn=xsgn0.upper()
    if xsgn=='1M':xsgn='1T'
    if xsgn=='5M':xsgn='5T'
    if xsgn=='15M':xsgn='15T'
    if xsgn=='30M':xsgn='30T'
    if xsgn=='60M':xsgn='60T'
    #
    return xsgn

   
    
def tq_timXGet(dfx,tim0,tim9):
    #
    timNum=len(dfx.index)
    if tim9=='':tim9=zdat.df_2ds(dfx,timNum).xtim 
    if tim0=='':tim0=zdat.df_2ds(dfx,1).xtim 
    #tim0,tim9=arrow.get(tim0),arrow.get(qx.btTim9str)
    #
    return tim0,tim9

def tick2x(df,ktim='1min'):
    '''
    ktim，是时间频率参数，请参看pandas的resample重新采样函数
        常见时间频率符号： 
            A， year 
            M， month 
            W， week 
            D， day 
            H， hour 
            T， minute 
            S，second
    '''
    #
    df['time']=pd.to_datetime(df['time']) 
    df=df.set_index('time')
    df=df.sort_index()
    #
    dfk=df['price'].resample(ktim).ohlc();dfk=dfk.dropna();
    vol2=df['volume'].resample(ktim).sum();vol2=vol2.dropna();
    df_vol2=pd.DataFrame(vol2,columns=['volume'])
    amt2=df['amount'].resample(ktim).sum();amt2=amt2.dropna();
    df_amt2=pd.DataFrame(amt2,columns=['amount'])
    #
    df2=dfk.merge(df_vol2,left_index=True,right_index=True)
    df9=df2.merge(df_amt2,left_index=True,right_index=True);
    #
    xtims=df9.index.format('%Y-%m-%d %H:%M:%S')
    del(xtims[0])
    df9['xtim']=xtims # df9.index.__str__();#  [str(df9.index)]
    #             
    return df9    

def xsyb2x(xsyb,kdiv='#'):
    x10=xsyb.split(kdiv)
    syb=x10[0]
    xsite,xsit9,syb9='','',syb
    #print(len(x10),x10)
    if len(x10)>1:
        xsite=x10[1]
        xsit9=xsite[0]+xsite[-1]
        syb9=syb+'_'+xsit9
    #
    return syb9,xsit9,syb,xsite
        
    
#-------tq.pools.xxxx
def tq_pools_init(qx):
    
    print('tq_pools init ...')
    qx.sybPools=zdat.pools_frd(qx.rdat,qx.sybLst,qx.priceSgn,qx.rdatTyp,qx.timFmtInv) #day:datType=''
    #
    syb=qx.sybLst[0]
    qx.wrkInx,qx.wrkInxDat=syb,qx.sybPools[syb]
    syb=qx.sybLst[1]
    qx.wrkSyb,qx.wrkSybDat=syb,qx.sybPools[syb]
    #
    qx.sybNum=len(qx.sybLst)

    #
    return qx
    
def tq_pools_wr(qx):
    fss=qx.rtmp+qx.wrkSybCod+'.csv'
    qx.wrkSybDat.to_csv(fss)
    
    
        
def tq_pools_chk(qx):
    print('\n@tq_pools_chk,xcode',qx.wrkSybCod)
    print(qx.wrkSybDat.tail())
    
def tq_pools_call(qx,vlst):
    #print('tq_pools call...')
    xfun,xtyp=vlst[0],vlst[1]
    #if xtyp==1:xlst,xpools=qx.inxLst,qx.inxPools
    #if xtyp==2:xlst,xpools=qx.sybLst,qx.sybPools
    xlst,xpools=qx.sybLst,qx.sybPools
    #
    for xcod in xlst:
        qx.wrkSybCod=xcod
        qx.wrkSybDat=xpools[xcod]
        #sta_dataPre(qx)
        xfun(qx)
        #
        xpools[xcod]=qx.wrkSybDat
        #
        #print('\ntq_pools_call,',xcod)
        #print(qx.sybPools[xcod].tail())
    #
    return qx
    



#---------------tq.trd.xxx

    

    
        
#---------------------------tq.syb.xxx
def tq_sybGetPrice(df,ksgn,xtim):
    '''
      获取当前价格
    
    Args:
        qx (zwQuantX): zwQuantX交易数据包
        ksgn (str): 价格模式代码
        '''
    #d10=dfw.sybLib[qx.sybCode]
    d01=df[xtim:xtim]
    #print(df.head());print('v',ksgn,xtim);print('d01',d01);
    #
    price=0;
    if len(d01)>0:
        d02=d01[ksgn]
        #print('d02',d02)
        price=d02.values[0];
        if pd.isnull(price):
            d02=d01['dprice']
            price=d02[0];
    #
    price=round(price,3)
    return price

#---------------------------syb

def syb2data_pre8FN(fss):
    if not os.path.exists(fss):
        return None
    #    
    df=pd.read_csv(fss,index_col=0)
    df['avg']=df[zsys.ohlcLst].mean(axis=1)
    #
    df['avg']=df[zsys.ohlcLst].mean(axis=1)
    df,avg_lst=zdat.df_xshift(df,ksgn='avg',num9=10)
    #print('avg_lst,',avg_lst)
    #
    mv_lst=[2,3,5,10,15,20,30,50,100,150,200]
    #ma_lst=[2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,80,100,120,150,180,200,250,300]
    df=zta.mul_talib(zta.MA,df, ksgn='avg',vlst=mv_lst)
    ma_lst=zstr.sgn_4lst('ma',mv_lst)
    #
    df['xtim']=df.index
    df['xyear']=df['xtim'].apply(zstr.str_2xtim,ksgn='y')
    df['xmonth']=df['xtim'].apply(zstr.str_2xtim,ksgn='m')
    df['xday']=df['xtim'].apply(zstr.str_2xtim,ksgn='d')
    df['xweekday']=df['xtim'].apply(zstr.str_2xtim,ksgn='w')
    tim_lst=['xyear','xmonth','xday','xweekday']
    #
    df['price']=df['avg']
    df['price_next']=df[avg_lst].max(axis=1)
    #涨跌幅,zsys.k_price_change=1000
    df['price_change']=df['price_next']/df['price']*100
    #df['ktype']=df['price_change'].apply(zt.iff2type,d0=100)  
    #def dat2type(d,k9=2000,k0=0):
    #fd>120
    #
    df=df.dropna()
    #df['ktype']=round(df['price_change']).astype(int)
    #df['ktype']=df['kprice'].apply(zt.iff2type,d0=100)  
    #df['ktype']=df['price_change'].apply(zt.iff3type,v0=95,v9=105,v3=3,v2=2,v1=1)  
    #
    df=df.round(3)
    return df


    
def syb2data_pre8Flst(finx,rss):
    flst=pd.read_csv(finx,index_col=False,dtype='str',encoding='gbk')
    df9=pd.DataFrame()
    xc=0
    for xcod in flst['code']:
        #print(xcod)
        xc+=1
        fss=rss+xcod+'.csv';print(xc,'#',fss)
        df=syb2data_pre8FN(fss)
        df9=df9.append(df)
    #
    return df9

#---------------------------user.xxx
def tq_usrIDSet(qx):
    ''' 生成订单流水号编码ID
       #ID=prjName+'_'+trdCnt(000000)
    '''

    qx.trdCnt+=1;
    nss='{:05d}'.format(qx.trdCnt);
    qx.trdID=qx.prjNam+'_'+nss;
    #

    return qx.trdID   

  

    

    