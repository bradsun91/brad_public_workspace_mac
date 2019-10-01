# -*- coding: utf-8 -*- 
'''
TopQuant-简称TQ极宽智能量化回溯分析系统，培训课件-配套教学python程序

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2017.10.1 首发


网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  
  
文件名:ztools_sta.py
默认缩写：import ztools_sta as zsta
简介：Top极宽量化·回溯策略模块
 

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




#-------------------
#-------------sta.talib.xxx.sub

#-------------sta.tools.trd
def sta_buy(qx,rx):
    stknum,dprice=0,qx.wrkPrice
    #print('\n@sta_buy:',stknum,dprice,qx.trd_mode)
    #
    if dprice>0:
        if qx.trd_mode==1:stknum=round(qx.trd_buyMoney/dprice)
        if qx.trd_mode==2:stknum=qx.trd_buyNum;
        dcash,dsum=qx.usrMoney,stknum*dprice
        #print('@sta_buy:',stknum,dcash+dsum,'@d',dsum,dcash)
        #
        if dsum>dcash:stknum=0
        #print('@sta_buy3:',stknum,dsum,dcash)
        
    #
    return stknum


def sta_sell(qx,rx):
    stknum,dprice=0,qx.wrkPrice
    usr_num9=qx.usr_num9
    #
    if usr_num9>0:
        ksell=1/qx.trd_sellNDiv
        qx.trd_sellNDiv=qx.trd_sellNDiv-1
        if qx.trd_sellNDiv<=0:qx.trd_sellNDiv=qx.trd_sellNDiv0
        stknum=round(usr_num9*ksell)
    #
    stknum=-stknum
    return stknum  

#-----------------
def trd_sub_chk(qx,fgTrd=0):
    stknum,dprice,ntim=qx.wrkSybNum,qx.wrkPrice,0
    #if fgTrd<0:print('@chk#1',stknum,dprice)
    if (qx.trd_tim!='')&(stknum>0):
        ntim=zt.timNHour(qx.wrkTimStr,qx.trd_tim)
        if ntim<qx.trd_NTimGap:stknum=0
    #    
    #if fgTrd<0:print('@chk#2',stknum,dprice,'@tn',ntim,qx.trd_NTimGap)
    if stknum!=0:
        dcash=qx.usrMoney
        qx.usrMoney=dcash-stknum*dprice
        qx.usr_num9=qx.usr_num9+stknum
        #
        qx.trd_tim=qx.wrkTimStr
        #
        #qx.usrBuyMoney0=qx.usrBuyMoney9
        qx.usrBuyMoney9=qx.usrBuyMoney9+stknum*dprice
        #
        if stknum<0:
            qx.trd_cnt+=1
    #
    #if fgTrd<0:print('@chk#3',stknum,dprice)
    return stknum

def trd_sub010(rx,vlst):  #第一个参数代表该函数处理的每一个元素，第二个参数args是传入的参数
    qx=vlst[0]
    #mbuy='>',msell='<'
    vbuy,mbuy,kbuy=vlst[1],vlst[2],vlst[3]
    vsell,msell,ksell=vlst[4],vlst[5],vlst[6]
    ksgn,stknum=qx.priceSgn,0
    qx.wrkTimStr,qx.wrkPrice=rx.xtim,rx[ksgn]
    #print('buy',vbuy,mbuy,kbuy); print('sell',vsell,msell,ksell)    #print(rx)
    #
    fgbuy,fgsell=False,False
    #
    if mbuy=='>':fgbuy=rx[vbuy]>kbuy
    if mbuy=='<':fgbuy=rx[vbuy]<kbuy
    if mbuy=='>$':fgbuy=rx[vbuy]>rx[kbuy]
    if mbuy=='<$':fgbuy=rx[vbuy]<rx[kbuy]
    #
    if msell=='>':fgsell=rx[vsell]>ksell
    if msell=='<':fgsell=rx[vsell]<ksell
    if msell=='>$':fgsell=rx[vsell]>rx[ksell]
    if msell=='<$':fgsell=rx[vsell]<rx[ksell]
    #
    if fgbuy:stknum=sta_buy(qx,rx)
    elif fgsell:stknum=sta_sell(qx,rx)
    #
    qx.wrkSybNum=stknum
    if stknum!=0:
        stknum=trd_sub_chk(qx) #xed.num9
        #
        #if rx.ktrd<0:print('@',ksgn,stknum)
    #
    
    if qx.usr_num9==0:
        vcash5,qx.usrBuyMoney9=0,0
    else:
        vcash5=qx.usrBuyMoney9/qx.usr_num9
    #    
    #print('@rx',rx)
    #if rx.ktrd!=0:print('\n@x',qx.trd_cnt,qx.usr_num9,stknum,qx.usrMoney,qx.usrBuyMoney9,vcash5    )
    #xxx
    return qx.trd_cnt,qx.usr_num9,stknum,qx.usrMoney,qx.usrBuyMoney9,vcash5    

def trd_sub(qx,df,vlst_sub):
    if not qx.trd_MulStaFlag:
        df.dropna(inplace=True)
        df=zdat.df_kcut8tim(df,'xtim',qx.btTim0str,'')
        #ucash0=df['ucash']
        df['rinx'],df['unum9'], df['unum'], df['ucash'], df['vcash9'],df['vcash5']=zip(*df.apply(trd_sub010, args =([qx]+vlst_sub,),axis=1)) 
        #
        df['vcash5dp']=df['close']
        df['vcash']=df['close']*df.unum9
        #df['vcash9']=df['vcash9']+df['vcash']
        #df['vcash9']=qx.usrBuyMoney9
        #df['vcash5']=df['vcash9']/df['unum9']
        df['vsum']=df.ucash+df.vcash
        
    #
    qx.wrkSybDat=df
    #
    return df

def trd_sub00(qx,df,vlst_sub):
    if not qx.trd_MulStaFlag:
        df.dropna(inplace=True)
        df=zdat.df_kcut8tim(df,'xtim',qx.btTim0str,'')
        #ucash0=df['ucash']
        x10=zip(*df.apply(trd_sub010, args =([qx]+vlst_sub,),axis=1)) 
        xnum=len(list(x10)) # print('\n@xn ',xnum)
        if xnum==6:
            [df['rinx'],df['unum9'], df['unum'], df['ucash'], df['vcash9'],df['vcash5']]=list(x10)
            #
            df['vcash5dp']=df['close']
            df['vcash']=df['close']*df.unum9
            #df['vcash9']=df['vcash9']+df['vcash']
            #df['vcash9']=qx.usrBuyMoney9
            #df['vcash5']=df['vcash9']/df['unum9']
            df['vsum']=df.ucash+df.vcash
        else:
            print('xxx trd_sub xnum',xnum)
            #xxx
        
    #
    qx.wrkSybDat=df
    #
    return df




#---------sta.misc.xxx
 
def Low10dp(qx):
    '''
    UpDown 鱼篓策略
    sta=[2,0,100,105] 
    sta=[10,0,100,105] 
    '''
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    #
    #df['dp1p']=df['dprice'].shift(v)
    df['dp1p']=df[ksgn].rolling(v).mean()
    df['kdp']=df['dprice']/df['dp1p']*100
    #zt.prDF('df',df,22);print(df.describe())
    #---sta.end.
    
    #trd_sub(qx,df,['kdp','<',v3, 'kdp','>',v4])
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.kdp<v3,'ktrd']=1
    df.loc[df.kdp>v4,'ktrd']=-1
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    return qx

def UpDown(qx):
    '''
    UpDown 鱼篓策略
    sta=[4,4,100,100] 
    sta=[30,30,104,101] 
    '''
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]   
    #
    df['dp1p']=df[ksgn].rolling(v).mean()
    df['kdp']=df['dprice']/df['dp1p']*100
    #
    df['dp_max0']=df['high'].rolling(v).max()
    df['dp_min0']=df['low'].rolling(v2).min()
    df['dmax1p']=df['dp_max0'].shift(1)
    df['dmin1p']=df['dp_min0'].shift(1)
    df['kmax']=df[ksgn]/df['dmax1p']*100
    df['kmin']=df[ksgn]/df['dmin1p']*100
    #zt.prDF('df',df,22);print(df.describe())
    #---sta.end.
    #trd_sub(qx,df,['kmin','<',v3, 'kmax','>',v4])
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.kmin<v3,'ktrd']=1
    df.loc[df.kmax>v4,'ktrd']=-1
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    return qx



#---------sta.ma.xxx      
def maAvg(qx):
    '''
    avg-MA均价均线策略
    sta=[10,0,105,105] 
    '''
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]
    #vnum=len(qx.staVars)
    #
    df['ma']=df[ksgn].rolling(v).mean()
    df['kavg']=df[ksgn]/df['ma']*100
    #zt.prDF('df',df,22);print(df.describe())
    #xxx
    #---sta.end.
    #sta_sub100(qx,df,tax2x_sub,['kavg','kavg',v2,v3])
    #trd_sub(qx,df,['kavg','<',v3, 'kavg','>',v4])
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.kavg<v3,'ktrd']=1
    df.loc[df.kavg>v4,'ktrd']=-1
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    return qx    
  
   
def maClose(qx):
    '''
    close-MA,close均线策略
    sta=[35,0,105,105] 
    '''
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    qx.priceSgn='close'
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]
    #vnum=len(qx.staVars)
    #
    df['ma']=df[ksgn].rolling(v).mean()
    df['kclose']=df[ksgn]/df['ma']*100
    #zt.prDF('df',df)
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.kclose<v3,'ktrd']=1
    df.loc[df.kclose>v4,'ktrd']=-1
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    
    return qx    


   
def ma2x(qx):
    '''
    MA2x 双均线策略
    sta=[80,5,105,105] 
    '''
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]
    #vnum=len(qx.staVars)
    #v,20
    df['ma_fast']=df[ksgn].rolling(v).mean()#
    df['ma_slow']=df[ksgn].rolling(v2).mean()
    df['kma']=df['ma_fast']/df['ma_slow']*100
    #zt.prDF('df',df,22);print(df.describe())
    #xxx
    #---sta.end.
    
    #sta_sub100(qx,df,tax2sgn_sub,['ma_fast','ma_slow','ma_slow','ma_fast'])
    #sta_sub100(qx,df,tax2x_sub,['kma','kma',v3,v4])
    #trd_sub(qx,df,['kma','<',v3, 'kma','>',v4])
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.kma<v3,'ktrd']=1
    df.loc[df.kma>v4,'ktrd']=-1
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #

    return qx    


   
def ma3x(qx):
    '''
    MA3x 三均线策略
    ma1,5 下穿 ma9,buy
    ma1,5 上穿 ma9,sell
    
    sta=[25,30,50,0] 
    
    '''
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]
    #vnum=len(qx.staVars)
    #v,20
    df['ma1']=df[ksgn].rolling(v).mean()#
    df['ma5']=df[ksgn].rolling(v2).mean()
    df['ma9']=df[ksgn].rolling(v3).mean()
    df['ma1p']=df['ma1'].shift(1)
    df['ma5p']=df['ma5'].shift(1)
    df['ma9p']=df['ma9'].shift(1)
    #df['kma']=df['ma_fast']/df['ma_slow']*100
    #
    df['ktrd']=0
    df.loc[(df.ma5>df.ma5p)&(df.ma1>df.ma1p)&(df.ma5>df.ma9)&(df.ma1>df.ma9),'ktrd']=1 #  buy
    df.loc[(df.ma5<df.ma5p)&(df.ma1<df.ma1p)&(df.ma5<df.ma9)&(df.ma1<df.ma9),'ktrd']=-1 #  sell
    #zt.prDF('df',df,22);print(df.describe())
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #

    return qx    


def maCross(qx):
    '''
    macross 均线交叉
    sta=[5,25,9,0] 
    '''
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3=qx.staVars[0],qx.staVars[1],qx.staVars[2] #,qx.staVars[3]
    #vnum=len(qx.staVars)
    #v,5+20; 8+34
    df['ma_cross']=0
    df['ma_fast']=df[ksgn].rolling(v).mean()#
    df['ma_slow']=df[ksgn].rolling(v2).mean()
    df['ma_fast2a']=df['ma_fast'].shift(v3)
    df['ma_slow2a']=df['ma_slow'].shift(v3)
    df.loc[(df.ma_fast>df.ma_fast2a)&(df.ma_fast>df.ma_slow)&(df.ma_fast2a<=df.ma_slow2a),'ma_cross']=1 # ma_up
    df.loc[(df.ma_fast<df.ma_fast2a)&(df.ma_fast<df.ma_slow)&(df.ma_fast2a>=df.ma_slow2a),'ma_cross']=-1 # ma_down
    #zt.prDF('df',df,22);print(df.describe())
    #xxx
    #
    df['ktrd']=0
    df.loc[df.ma_cross<0,'ktrd']=1 
    df.loc[df.ma_cross>0,'ktrd']=-1 
    #zt.prDF('df',df,22);print(df.describe())
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    #
    return qx    

#------a--z
    
def ACCDIST(qx):
    '''   
    AD集散指标策略
    集散指标(A/D)——Accumulation/Distribution
        也称离散指标，是由价格和成交量的变化而决定的
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：ad_{n}，输出数据
        若A/D指标上升，而价格下降时，为买进信号。
        若A/D指标下降，而价格上升，为卖出信号；
        
    默认参数示例：
    qx.staVars=[2,4,102,102]     
 
'''    
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]
    #
    df=zta.ACCDIST(df,v,ksgn)
    df['dp_n']=df['dprice'].rolling(center=False,window=v2).mean()*v3/100 
    df['ad_n']=df['ad'].rolling(center=False,window=v2).mean()*v4/100 
    #
    #df['xma']=df['vortex'].rolling(center=False,window=v2).mean() 
    #df['x1p']=df['xma'].shift(1)
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[(df.dprice<df.dp_n)&(df.ad>df.ad_n),'ktrd']=1
    df.loc[(df.dprice>df.dp_n)&(df.ad<df.ad_n),'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xxx2
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    
    return qx    

  

   
def ADX(qx):
    '''   
    ADX平均趋向策略
    def ADX(df, n, n_ADX):
    adx，中文全称：平均趋向指数，ADX指数是反映趋向变动的程度，而不是方向的本身
    英文全称：Average Directional Index 或者Average Directional Movement Index
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        n_ADX,adx周期
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：adx_{n}_{n2}，输出数据
    ADX 读数上升，代表趋势转强；如果 ADX 读数下降，意味着趋势转弱。
    ADX 读数为 30 或以上（参考图 6-8 ） ，趋势就可以视为强劲。如果 ADX 读数低于 20 ,代表市场动能偏弱
    
    默认参数示例：
    qx.staVars=[8,30,20,30]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    
    #
    df=zta.ADX(df,v,v2,ksgn)
    #
    df['dp1p']=df['dprice'].shift(1)
    df['adx1p']=df['adx'].shift(1)
    df['kdp']=df['dprice']/df['dp1p']*100 
    df['kadx']=df['adx']/df['adx1p']*100 
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[(df.kdp>100)&(df.kadx>100)&(df.adx>v3),'ktrd']=1
    df.loc[(df.kdp<100)&(df.kadx<100)&(df.adx>v3),'ktrd']=-1
    df.loc[(df.kdp>100)&(df.kadx>100)&(df.adx<v4),'ktrd']=-1
    df.loc[(df.kdp<100)&(df.kadx<100)&(df.adx<v4),'ktrd']=1
    #df.loc[(df.kdp<v3)&(df.kadx<v4),'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xxx2
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    return qx          

 

def ATR(qx):
    '''   
    ATR,均幅指标策略
   ATR,均幅指标（Average True Ranger）,取一定时间周期内的股价波动幅度的移动平均值，主要用于研判买卖时机
    
   海龟交易法则: @dp<0.5ATR,buy;  @dp>2ATR,sell
   
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：atr_{n}，输出数据
    默认参数示例：
    qx.staVars=[35,0,100,95]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    #
    df=zta.ATR(df,v,ksgn)
    #df['katr']=df['dprice']/df['atr']*100
    #zt.prDF('df',df)
    #xx
    #
    df['dbuy']=df['atr']*v3/100
    df['dsell']=df['atr']*v4/100
    #
    df['ktrd']=0
    df.loc[df.dprice>df.dbuy,'ktrd']=1 #buy
    df.loc[df.dprice<df.dsell,'ktrd']=-1 #sell
    #zt.prDF('df',df,22);print(df.describe())
    # ccc
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #sta_sub100(qx,df,ATR10_sub,[v2,v3])
    #sta_sub100(qx,df,tax2x_sub,['katr','katr',v2,v3])
    #
    
    return qx    


def BBANDSX(qx):
    '''
    zw改进版布林带策略:BBANDS_UpLow
    Bollinger Bands  
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了4栏：
            boll_ma，布林带均线数据
            boll_std，布林带方差据
            boll_up，布林带上轨带差据
            boll_low，布林带下轨带差据
    默认参数示例：
    qx.staVars=[15,0,100,102]     
        '''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]          
    #
    df=zta.BBANDS_UpLow(df,v,ksgn)
    #df['boll_low']=df['boll_low']*v2/100
    #df['boll_up']=df['boll_up']*v3/100
    #
    df['dbuy']=df['boll_low']*v3/100
    df['dsell']=df['boll_up']*v4/100
    #
    df['ktrd']=0
    df.loc[df.dprice<df.dbuy,'ktrd']=1 #buy
    df.loc[df.dprice>df.dsell,'ktrd']=-1 #sell
    #zt.prDF('df',df,22);print(df.describe())
    # ccc
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    
    #
    return qx      
    
    
def CCI(qx):
    '''   
    CCI 策略
     CCI顺势指标(Commodity Channel Index)
    CCI指标，是由美国股市分析家唐纳德·蓝伯特（Donald Lambert）所创造的，是一种重点研判股价偏离度的股市分析工具。

    
    MA是简单平均线，也就是平常说的均线
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：cci，输出数据
        
    默认参数示例：
    qx.staVars=[25,0,100,100]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.CCI(df,v,ksgn)
    #df['cci']=df['cci']*100
    #zt.prDF('df',df)
    #xx
    #
    #
    df['dbuy']=df['cci']*v3
    df['dsell']=df['cci']*v4
    #
    df['ktrd']=0
    df.loc[df.dprice<df.dbuy,'ktrd']=1 #buy
    df.loc[df.dprice>df.dsell,'ktrd']=-1 #sell
    #zt.prDF('df',df,22);print(df.describe())
    # ccc
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    
    #
    
    return qx    

    
def COPP(qx):
    '''   
    COPP 估波指标 策略
 　　估波指标（Coppock Curve）又称“估波曲线”，通过计算月度价格的变化速率的加权平均值来测量市场的动量，属于长线指标。
　　估波指标由Edwin·Sedgwick·Coppock于1962年提出，主要用于判断牛市的到来。
    该指标用于研判大盘指数较为可靠，一般较少用于个股；再有，该指标只能产生买进讯号。
    依估波指标买进股票后，应另外寻求其他指标来辅助卖出讯号。
    估波指标的周期参数一般设置为11、14，加权平均参数为10，也可以结合指标的平均线进行分析

    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：copp_{n}，输出数据
        
    默认参数示例：#11,0,14,10
    qx.staVars=[60,0,5,5]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True)
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.COPP(df,v,ksgn)
    df['copp']=df['copp']*100
    #df['dbuy']=df['copp']*100
    #df['dsell']=df['copp']*100
    #
    df['ktrd']=0
    df.loc[df.copp<v3,'ktrd']=1 #buy
    df.loc[df.copp>v4,'ktrd']=-1 #sell
    #zt.prDF('df',df,22);print(df.describe())
    #ccc
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    return qx    

    
def CHAIKIN(qx):
    '''   
    CHAIKIN,佳庆指标策略
　　佳庆指标（CHAIKIN,Chaikin Oscillator）是由马可·蔡金（Marc Chaikin）提出的，聚散指标（A/D）的改良版本。
    【输入】
        df, pd.dataframe格式数据源
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：ck，输出数据
 　　佳庆指标由负值向上穿越0轴时，为买进讯号。（注意！股价必须位于90天移动平均线之上，才可视为有效）。
   佳庆指标由正值向下穿越0轴时，为卖出讯号。（注意！股价必须位于90天移动平卷线之下，才可视为有效）。   
    默认参数示例：
    qx.staVars=[6,0, 95, 95]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    #
    df=zta.CHAIKIN(df,v,ksgn)
    df['ma90']=zta.MA01(df,90,ksgn)
    df['ck1p']=df['ck'].shift(1)
    #
    df['dbuy']=df['ma90']*v3/100
    df['dsell']=df['ma90']*v4/100
    #
    df['ktrd']=0
    df.loc[(df.dprice>df.dbuy)&(df.ck>0)&(df.ck1p<0),'ktrd']=1 #buy
    df.loc[(df.dprice<df.dbuy)&(df.ck<0)&(df.ck1p>0),'ktrd']=-1 #sell
    #zt.prDF('df',df,22);print(df.describe())
    #ccc
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    
    #
    return qx   

   
def FibPR(qx):
    '''   
    ??? FibPR斐波纳契策略
        Fibonacci Price Retracements 
    斐波纳契价格回调(黄金分割线)指数
    当 price<fib-381,-618,买入
    当 price>fib+381,+618,，卖出
    默认参数示例：
    qx.staVars=[0,5,100,50]     
 
'''    
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2] ,qx.staVars[3]
    #
    df=zta.FibPR(df,ksgn)
    #rlst=['fib-618','fib-381','fib381','fib618']
    #df=zta.tax_shift(df,rlst,1)
    df['dbuy0']=df['fib381'].rolling(center=False,window=v2).mean() 
    df['dsell0']=df['fib-381'].rolling(center=False,window=v2).mean() 
    df['dbuy']=df['dbuy0']*v3/100
    df['dsell']=df['dsell0']*v4/100
    #xxx
    df['ktrd']=0
    df.loc[df.dprice<df.dbuy,'ktrd']=1 #buy
    df.loc[df.dprice>df.dsell,'ktrd']=-1 #sell
    #zt.prDF('df',df,22);print(df.describe())
    # ccc
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
        
    #

    return qx    
                         

def KDJ(qx):
    '''   
     KDJ策略10
     KDJ 指标，又称随机指标
    当 stok>90，买入；??
    当 stok<10，卖出
    默认参数示例：
    qx.staVars=[12,0,10,60]    [9,15,70]    
 
'''    
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]
    #
    df=zta.STOD(df,v,ksgn);
    #zt.prDF('df',df,22);print(df.describe())
    #zzz
    #
    df['ktrd']=0
    df.loc[df.stok>v3,'ktrd']=1 #buy
    df.loc[df.stok<v4,'ktrd']=-1 #sell
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    
    return qx    





def KDJ20(qx):
    '''   
 def KDJ(df, n,ksgn='close'):     
       KDJ 随机指标D值,Stochastic oscillator %D  
	随机指标，又称KD指标，KDJ指标
    根据目前股价在近阶段股价分布中的相对位置来预测可能发生的趋势反转
　   随机指标综合了动量观念、强弱指标及移动平均线的优点，用来度量股价脱离价格正常范围的变异程度。
　   KD指标考虑的不仅是收盘价，而且有近期的最高价和最低价，这避免了仅考虑收盘价而忽视真正波动幅度的弱点。
　  随机指标一般是根据统计学的原理，通过一个特定的周期（常为9日、9周等）内出现过的最高价、最低价
  及最后一个计算周期的收盘价及这三者之间的比例关系，来计算最后一个计算周期的未成熟随机值RSV，
  然后根据平滑移动平均线的方法来计算K值、D值与J值，并绘成曲线图来研判股票走势。
  K与D值永远介于0到100之间。D大于80时，行情呈现超买现象。D小于20时，行情呈现超卖现象。
  上涨趋势中，K值小于D值，K线向上突破D线时，为买进信号。下跌趋势中，K值大于D值，K线向下跌破D线时，为卖出信号。


算法
对每一交易日求RSV（未成熟随机值）
RSV=（收盘价-最近N日最低价）/（最近N日最高价-最近N日最低价）* 100
K线：RSV的M1日移动平均
D线：K值得M2日移动平均
J线：3D-2K。
参数
N，M1 。M2 天数，一般为9，3，3
用法
（1）D大于80 ，超买； D小于20，超卖；J大于100，超买；J小于10(or 0)，超卖。
 （2）指标线K向上突破指标线D，买进信号；指标线K向下跌破指标线D，卖出信号。
 （3）指标线K与指标线D的交叉发生在70以上和30以下才有效。、
 （4）KDJ指标不适于发行量小，交易不活跃的股票。
 （%）KDJ指标对于大盘和热门大盘股有极高准确性。
 
       
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：stod，输出数据
 　　   
    默认参数示例：
    qx.staVars=[21,0,0,100]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.KDJ(df,v,ksgn)
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    #df.loc[(df.roc>0)&(df.roc>df.roc1p),'ktrd']=1
    #df.loc[(df.roc<0)&(df.roc<df.roc1p),'ktrd']=-1
    df.loc[df.kdj_d<20,'ktrd']=1
    df.loc[df.kdj_d>80,'ktrd']=-1
    df.loc[df.kdj_j<v3,'ktrd']=1
    df.loc[df.kdj_d>v4,'ktrd']=-1
    #
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    

    
    return qx   

  


def KELCH(qx):
    '''   
    KELCH 肯特纳通道 策略
     肯特纳通道（Keltner Channel，KC）,是一个移动平均通道，由叁条线组合而成(上通道、中通道及下通道)。
	KC通道，一般情况下是以上通道线及下通道线的分界作为买卖的最大可能性。
  	若股价於边界出现不沉常的波动，即表示买卖机会。   
    @ 当价格报收在顶部环带之上时，意味着价格呈强势，后市看涨。
　　@当价格报收在底部环带之下时，意味着价格呈弱势，后市看跌。
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了3栏：kc_m，中间数据
            kc_u，up上轨道数据
            kc_d，down下轨道数据
 　　   
    默认参数示例：
    qx.staVars=[150,0,90,100]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    #
    df=zta.KELCH(df,v,ksgn)
    df['kbuy']=df['dprice']/df['kc_u']*100
    df['ksell']=df['dprice']/df['kc_d']*100
    #zt.prDF('df',df,22)
    #xx
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.kbuy>v3,'ktrd']=1
    df.loc[df.ksell<v4,'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    
    return qx    
    

def DONCH(qx):
    '''   
    DONCH,奇安通道策略 tur,n=20
　　奇安通道指标,Donchian Channel  
	该指标是由Richard Donchian发明的，是有3条不同颜色的曲线组成的，该指标用周期（一般都是20）内的最高价和最低价来显示市场的波动性
	当其通道窄时表示市场波动较小，反之通道宽则表示市场波动比较大。
   @tur 海龟策略 
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        
    【输出】    
        df, pd.dataframe格式数据源,
        增加了2栏：donchsr，中间输出数据
            donch，输出数据
 　　
    突破20日最高价，buy
    跌破20日最低价，sell   
    默认参数示例：
    qx.staVars=[3,0,95,105]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.DONCH_ext(df,v)
    df['dc_up']=df['dc_up']*v3/100
    df['dc_low']=df['dc_low']*v4/100
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    #
    #trd_sub(qx,df,['dprice','<$','dc_low', 'dprice','>$','dc_up'])
    #
    df['ktrd']=0
    df.loc[df.dprice<df.dc_low,'ktrd']=1 #buy
    df.loc[df.dprice>df.dc_up,'ktrd']=-1 #sell
    #zt.prDF('df',df,22);print(df.describe())
    #ccc
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    
    #
    return qx    
 
def EMA(qx):
    '''   
    EMA指数均线策略
    #Exponential Moving Average  
    EMA是指数平滑移动平均线，也叫EXPMA指标，也称为：SMMA 
    是平均线的一个变种，EMA均线较MA更加专业一些。
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：ema，输出数据
 　　当短天期天数线下穿长天期天数线时，buy
   当短天期天数线上穿长天期天数线时，sell
    默认参数示例：
    qx.staVars=[10,20,0,0]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    #
    df2=zta.EMA_ext(df,v,ksgn)
    df['ema1']=df2['ema']
    df2=zta.EMA_ext(df,v2,ksgn)
    df['ema9']=df2['ema']
    df['ema1p']=df['ema1'].shift(1)
    df['ema9p']=df['ema9'].shift(1)
    #
    #
    df['ktrd']=0
    df.loc[(df.ema1>df.ema1p)&(df.ema1>df.ema9),'ktrd']=1  #buy
    df.loc[(df.ema1<df.ema1p)&(df.ema1<df.ema9),'ktrd']=-1 #sell
    #zt.prDF('df',df,22);print(df.describe())
    #ccc
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    

    #
    
    return qx    


def EOM(qx):
    '''   
    EOM 简易波动策略
      简易波动指标(Ease of Movement Value)，又称EMV指标
   它是由RichardW．ArmJr．根据等量图和压缩图的原理设计而成,目的是将价格与成交量的变化结合成一个波动指标来反映股价或指数的变动状况。
   由于股价的变化和成交量的变化都可以引发该指标数值的变动,因此,EMV实际上也是一个量价合成指标。
   当EOM由下往上穿越0轴时，买进。  当EOM由上往下穿越0轴时，卖出。

    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了2栏：eom_{n}，输出数据
            eom_x，10e10倍的输出数据
 　　   
    默认参数示例：
    qx.staVars=[12,0,120,100]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    #
    df=zta.EOM(df,v)
    df['eom1p2']=df['eom_x'].shift(1)*v3/100
    df['eom1p3']=df['eom_x'].shift(1)*v4/100
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[(df.eom_x>0)&(df.eom_x>df.eom1p2),'ktrd']=1
    df.loc[(df.eom_x<0)&(df.eom_x<df.eom1p3),'ktrd']=-1
    #
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    
    return qx    
                 
def FORCE(qx):
    '''   
    FORCE 劲道指数 策略
    劲道指数(Force Index),是由亚历山大·埃尔德(Alexander Elder)博士设计的一种摆荡指标，藉以衡量每个涨势中的多头劲道与每个跌势中的空头劲道。
　　劲道指数结合三项主要的市场资讯：价格变动的方向、它的幅度与成交量。它是由一个崭新而实用的角度，把成交量纳入交易决策中。
  短线的交易，则在劲道指数翻为正值时卖出，在劲道指数翻为负值时回补
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了2栏：force__{n}，输出数据
          force_x，缩小10e7倍的输出数据
 　　   
    默认参数示例：
    qx.staVars=[60,0,80,80]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    #
    df=zta.FORCE(df,v,ksgn)
    df['force2']=df['force'].shift(1)*v3/10000
    df['force3']=df['force'].shift(1)*v4/10000
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[(df.force<0)&(df.force<df.force2),'ktrd']=1
    df.loc[(df.force>0)&(df.force>df.force3),'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    
    #
    
    return qx    



    
def MACD(qx):
    '''   
    MACD策略02
     MACD称为指数平滑异同平均线
    当 macd>macd_sign，买入；
    当 macd<macd_sign，卖出
    默认参数示例：
    qx.staVars=[12,26]   
    qx.staVars=[20,10,100,100]    
'''    
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2] ,qx.staVars[3]
    #
    df=zta.MACD(df,v,v2,ksgn)
    df['macd']=df['macd']*1000
    #df['msign']=df['msign']*100
    df['kbuy']=df['msign']*v3*10
    df['ksell']=df['msign']*v4*10
    #zt.prDF('df',df,22);print(df.describe())
    #xxx
    df['ktrd']=0
    df.loc[df.macd>df.kbuy,'ktrd']=1 #buy
    df.loc[df.macd<df.ksell,'ktrd']=-1 #sell
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    return qx    
  
    
def MFI(qx):
    '''   
    MFI 金流量指标策略
    
    MFI,资金流量指标和比率,Money Flow Index and Ratio
　　资金流量指标又称为量相对强弱指标（Volume Relative Strength Index，VRSI），
	英文全名Money Flow Index，缩写为MFI，根据成交量来计测市场供需关系和买卖力道。
	该指标是通过反映股价变动的四个元素：上涨的天数、下跌的天数、成交量增加幅度、成交量减少幅度
	来研判量能的趋势，预测市场供求关系和买卖力道，属于量能反趋向指标。	    
    @当MFI>80，而产生背离现象时，视为卖出信号。
    @.当MFI<20，而产生背离现象时，视为买进信号
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：mfi，输出数据
 　　   
    默认参数示例：
    qx.staVars=[14,0,20,80]  
    [50,0,100,100]
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    #
    df=zta.MFI(df,v,ksgn)
    #zt.prDF('df',df,22)
    #xx
    #
    #trd_sub(qx,df,['mfi','<',v3, 'mfi','>',v4])
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.mfi<3,'ktrd']=1
    df.loc[df.mfi>v4,'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    
    return qx    


def MOM(qx):
    '''   
    MOM 动量线 策略
    动量线，英文全名MOmentum，简称MOM。“动量”这一名词，市场上的解释相当广泛。以Momentum命名的指标，种类更是繁多。
		综合而言，动量可以视为一段期间内，股价涨跌变动的比率。
    
    动量指标.Momentum  
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：mom，输出数据
        
    12天M0M以O轴为中心线，O轴的上、下方，分成六等份的超买超卖区，分别为＋1、＋2、+3和-1、-2、-3。
    1、短线行情，12日MOM上升至+1时，股价回档。
　　2.短线行情，12日MOM下跌至-1时，股价反弹。
　　3.中期趋势， 2日MOM＞＋2时，经常是上升波段结束的时机。
　　4.中期趋势，12日MOM＜-2时，经常是下跌波段结束的时机。
  ---
    1、25天MOM＞O轴，代表中期多头走势。
　　2.25天MOM＜O轴，代表中期空头走势。
 　　   
    默认参数示例：
    qx.staVars=[5,0,30,50]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.MOM(df,v,ksgn)
    df['mom']=df['mom']*50
    #zt.prDF('df',df,22); print(df.describe())
    #xx
    #
    #trd_sub(qx,df,['mom','>',v3, 'mom','<',-v4])
    #sta_sub100(qx,df,tax2x_sub,['mom','mom',v2,v3])
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.mom>v3,'ktrd']=1
    df.loc[df.mom<-v4,'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    
    return qx    

def MASS(qx):
    '''   
    MASS 梅斯线 策略
     def MassI(df):					
    梅斯线（Mass Index）
　　梅斯线是Donald Dorsey累积股价波幅宽度之后，所设计的震荡曲线。
		本指标最主要的作用，在于寻找飙涨股或者极度弱势股的重要趋势反转点。
　　MASS指标是所有区间震荡指标中，风险系数最小的一个。		
    
    【输入】
        df, pd.dataframe格式数据源
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：mass，输出数据
 　　   
    参数设定为25天。可视需要缩短周期至12天。
    1、ASS曲线向上穿越27,随后又掉头跌落26.5。当时，如果股价的9天移动平均线，正处于上升状态，代表多头行情即将反转下跌。
　　2、MASS曲线向上穿越27,随后又掉头跌落26.5。当时，如果股价的9天移动平均线，正处于下跌状态，代表空头行情即将反转上涨。
　　3、MASS曲线低于25的股票，一般不具有投资机会。
  将梅斯线参数由默认值25改为60，搭配的移动平均线由9改为25，可过滤掉一些不必要的信号，提供准确性
    
    默认参数示例：
    qx.staVars=[]     
   [25,9,8,0]
   [60,25,8,0]
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.MASS(df,v)
    df=zta.MA(df,v2,ksgn)
    df['ma1p']=df['ma'].shift(1)
    df['m_high']=df['mass'].rolling(center=False,window=v3).max() 
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    k6,k7=26.5,27
    df.loc[(df.mass<k6)&(df.m_high>k7)&(df.ma>df.ma1p),'ktrd']=-1
    df.loc[(df.mass<k6)&(df.m_high>k7)&(df.ma<df.ma1p),'ktrd']=1
    #
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    
    #
    
    return qx    


def OBV(qx):
    '''   
    OBV 能量潮策略
        #能量潮指标（On Balance Volume，OBV）
    OBV指标是葛兰维（Joe Granville）于本世纪60年代提出的，并被广泛使用。
    股市技术分析的四大要素：价、量、时、空。OBV指标就是从“量”这个要素作为突破口，来发现热门股票、分析股价运动趋势的一种技术指标。
    它是将股市的人气——成交量与股价的关系数字化、直观化，以股市的成交量变化来衡量股市的推动力，从而研判股价的走势。
    关于成交量方面的研究，OBV能量潮指标是一种相当重要的分析指标之一。    
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了2栏：obv，输出数据
        obv_x，放大10e6倍的输出数据
 　　   
    默认参数示例：
    qx.staVars=[5,5,0,0]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.OBV(df,v,ksgn)
    df['obvx']=df['obv'].rolling(center=False,window=v2).mean() 
    df['obv1p']=df['obvx'].shift(1)
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[(df.obv>0)&(df.obv1p<0),'ktrd']=1
    df.loc[(df.obv<0)&(df.obv1p>0),'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    
    return qx    



                        
def PPSR(qx):
    '''   
    PPSR策略
         支点，支撑线和阻力线.Pivot Points, Supports and Resistances  
	      PIVOT指标的观念很简单，不需要计算任何东西，它纯粹只是一个分析反转点的方法而已。
	     PIVOT意思是指“轴心”，轴心是用来确认反转的基准，所以PIVOT指标其实就是找轴心的方法
        PIVOT指标，经常与布林带数据一起分析。
    当 price>r1,r2，卖出
    当 price<s1,s2，买入
    默认参数示例：
    qx.staVars=[0,0,100,105,]     
'''    
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]
    #
    df=zta.PPSR(df)
    rlst=['r1','r2','s1','s2']
    #r3lst=['pp','r3','s3']
    df=zta.tax_shift(df,rlst,1)
    df['dbuy']=df['s2p1']*v3/100
    df['dsell']=df['r2p1']*v4/100
    #
    #df=zta.tax_shift(df,rlst,1)
    #df['dbuy0']=df['fib381'].rolling(center=False,window=v2).mean() 
    #df['dsell0']=df['fib-381'].rolling(center=False,window=v2).mean() 
    #df['dbuy']=df['dbuy0']*v3/100
    #df['dsell']=df['dsell0']*v4/100
    #xxx
    df['ktrd']=0
    df.loc[df.dprice<df.dbuy,'ktrd']=1 #buy
    df.loc[df.dprice>df.dsell,'ktrd']=-1 #sell
    #zt.prDF('df',df,22);print(df.describe())
    # ccc
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    
    return qx    

def ROC(qx):
    '''   
    ROC 变动率 策略
    变动率(Rate of change,ROC)
　　ROC是由当天的股价与一定的天数之前的某一天股价比较，其变动速度的大小,来反映股票市场变动的快慢程度。
		ROC，也叫做变动速度指标、变动率指标或变化速率指标。
    
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：roc，输出数据
 　　   
    @采用12天及25天周期可达到相当的效果    
    ROC自上而下跌破0,是卖出信号。反之，ROC自下而上穿过0,是买进信号。
    ROC上穿ROCAVG并且ROC为正值时,是买入信号。同理，ROC下穿ROCAVG并且ROC为负值时，是卖出信号。
    
    默认参数示例：
    qx.staVars=[3,15,9,0]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.ROC(df,v,ksgn)
    df=zta.MA(df,v2,'roc')
    df.rename(columns={'ma' : 'mroc'},inplace=True) 
    df['rocx']=df['roc'].rolling(center=False,window=v3).mean() 
    df['roc1p']=df['rocx'].shift(1)
    df['ktrd']=0  # =1:buy; =-1:sell
    #df.loc[(df.roc>0)&(df.roc>df.roc1p),'ktrd']=1
    #df.loc[(df.roc<0)&(df.roc<df.roc1p),'ktrd']=-1
    df.loc[(df.roc>0)&(df.roc>df.roc1p),'ktrd']=1
    df.loc[(df.roc<0)&(df.roc<df.roc1p),'ktrd']=-1
    df.loc[(df.roc>0)&(df.roc<df.mroc),'ktrd']=1
    df.loc[(df.roc<0)&(df.roc>df.mroc),'ktrd']=-1
    
    #df.loc[(df.obv<0)&(df.obv1p>0),'ktrd']=-1
    
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    
    return qx    


def RSI(qx):
    '''   
    RSI策略
    RSI相对强弱指标
    当 rsi>kbuy，一般是70,80，买入
    当 rsi<sell，一般是30，20，卖出
    默认参数示例：
    qx.staVars=[10,0,30,55]     
 
'''    
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]
    #
    df=zta.RSI(df,v);
    #zt.prDF('df',df,22);print(df.describe())
    #zzz
    #
    df['ktrd']=0
    df.loc[df.rsi>v3,'ktrd']=1 #buy
    df.loc[df.rsi<v4,'ktrd']=-1 #sell
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #

    return qx    


def TRIX(qx):
    '''   
    TRIX 三重指数均线 策略
    
    三重指数平滑平均线（TRIX）属于中长线指标。它过滤掉许多不必要的波动来反映股价的长期波动趋势。
    在使用均线系统的交叉时，有时会出现骗线的情况，有时还会出现频繁交叉的情况，通常还有一个时间上的确认。
    为了解决这些问题，因而发明了TRIX这个指标把均线的数值再一次地算出平均数，并在此基础上算出第三重的平均数。这样就可以比较有效地避免频繁出现交叉信号。 TRIX指标又叫三重指数平滑移动平均指标，其英文全名为“Triple Exponentially Smoothed Average”，是一种研究股价趋势的长期技术分析工具。
    
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：trix，输出数据
    
 　　参数N设为12，参数M设为20；   
    默认参数示例：
    qx.staVars=[20,15,15,0]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]  
    #
    df=zta.TRIX(df,v,ksgn)
    df['tma']=zta.MA01(df,v2,'trix')
    df['xtrix']=df['trix'].rolling(center=False,window=v3).mean() 
    df['xtrix1p']=df['xtrix'].shift(1)
    
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[(df.trix>df.tma)&(df.trix>df.xtrix1p),'ktrd']=1
    df.loc[(df.trix<df.tma)&(df.trix<df.xtrix1p),'ktrd']=-1
    #
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    
    return qx    


def TSI(qx):
    '''   
    TSI 真实强度 策略
    TSI，真实强度指数,True Strength Index
  TSI是相对强弱指数 (RSI) 的变体。
  TSI 使用价格动量的双重平滑指数移动平均线，剔除价格的震荡变化并发现趋势的变化。
  r一般取25，是一般取13
    【输入】
        df, pd.dataframe格式数据源
        r,s，时间长度;  r一般取25，s一般取13
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：tsi，输出数据
 　　   
    默认参数示例：
    qx.staVars=[25,15,3,25]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.TSI(df,v,v2,ksgn)
    df['tsi']=df['tsi']*100
    df['xma']=df['tsi'].rolling(center=False,window=v3).mean() 
    df['x1p']=df['xma'].shift(1)
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[(df.tsi>-v4)&(df.tsi>df.x1p),'ktrd']=1
    df.loc[(df.tsi<v4)&(df.tsi<df.x1p),'ktrd']=-1
    
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    return qx    

      
   
            
  
def Tur10(qx):
    '''     海龟策略:tur10
    tur10 :
    当今天的收盘价，大于过去n个交易日中的最高价时，以收盘价买入；
    买入后，当收盘价小于过去n个交易日中的最低价时，以收盘价卖出。
    
    
    tur20 :  海龟交易法则: @dp<0.5ATR,buy;  @dp>2ATR,sell
    默认参数示例：
    
    sta=[5,5,100,102] 
    sta=[30,30,102,102] 
    
        '''    
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3]
    #
    df['dprice']=df[ksgn]
    df['xhigh0']=df[ksgn].rolling(window=v,center=False).max()
    df['xlow0']=df[ksgn].rolling(window=v2,center=False).min()
    df['xhigh']=df['xhigh0'].shift(1)*v3/100
    df['xlow']=df['xlow0'].shift(1)*v4/100
    #zt.prDF('df',df,22);print(df.describe())
    #---sta.end.
    #sta_sub100(qx,df,tax2sgn_sub,['dprice','dprice','xhigh','xlow'])
    #trd_sub(qx,df,['dprice','<$','xlow', 'dprice','>$','xhigh'])
    #
     #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.dprice<df.xlow,'ktrd']=1
    df.loc[df.dprice>df.xhigh,'ktrd']=-1
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #

    return qx      


def UOS(qx):
    '''   
    UOS，终极指标 策略
     UOS，终极指标（Ultimate Oscillator，UOS）
　　终极指标，由拉瑞·威廉（Larry Williams）所创。他认为现行使用的各种振荡指标，对于周期参数的选择相当敏感。
   不同的市况，不同参数设定的振荡指标，产生的结果截然不同。因此，选择最佳的参数组含，成为使用振荡指标之前，最重要的一道手续。
　　为了将参数周期调和至最佳状况，拉瑞·威廉经过不断测试的结果，先找出三个周期不同的振荡指标，再将这些周期参数，按照反比例的方式，制作成常数因子。
   然后，依照加权的方式，将三个周期不同的振荡指标，分别乘以不同比例的常数，加以综合制作成UOS指标。
　　经过一连串参数顺化的过程后，UOS指标比一般单一参数的振荡指标，更能够顺应各种不同的市况。
    【输入】
        df, pd.dataframe格式数据源
        ksgn，列名，一般是：close收盘价
    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：uos，输出数据
 　　   
     uos短线抄底：uos上穿50
     uos短线卖顶：uos下穿65
     uos中长期抄底：uos上穿35
     uos中长期卖顶：uos下穿70   
     @买入，短期UOS上穿65；uos中长期抄底：uos上穿35
     @卖出，短期UOS下穿65；uos中长期卖顶：uos下穿70   
     
    默认参数示例：@ n=7
    qx.staVars=[7,20,85,85]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2],qx.staVars[3] 
    #
    df=zta.ULTOSC(df,v,ksgn)
    #df['tsi']=df['tsi']*100
    df['xma']=df['uos'].rolling(center=False,window=v2).mean() 
    df['x1p']=df['xma'].shift(1)
    df['ktrd']=0  # =1:buy; =-1:sell
    #
    df.loc[(df.uos>v3)&(df.uos>df.x1p),'ktrd']=1
    df.loc[(df.uos<v4)&(df.uos<df.x1p),'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    #
    
    return qx    




def VORTEX(qx):
    '''   
    VORTEX 螺旋指标 策略
    def VORTEX(df, n):
    VI螺旋指标,#Vortex Indicator  
    参见 http://www.vortexindicator.com/VFX_VORTEX.PDF


    
    【输入】
        df, pd.dataframe格式数据源
        n，时间长度

    【输出】    
        df, pd.dataframe格式数据源,
        增加了一栏：vortex 输出数据
 　　   
    默认参数示例：
    qx.staVars=[7,7,15,15]     
 
'''    
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2] ,qx.staVars[3] 
    #
    df=zta.VORTEX(df,v)
    #
    df['xma']=df['vortex'].rolling(center=False,window=v2).mean() 
    df['x1p']=df['xma'].shift(1)
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[(df.vortex>v3),'ktrd']=1
    df.loc[(df.vortex<-v4),'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xx
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    
    return qx    



  
def VWAP(qx):
    '''
    vwap策略，成交量加权平均价
    sta=[10,10,15,5] 
    '''
    df,ksgn=qx.wrkSybDat,qx.priceSgn
    df=df.sort_index(ascending=True);
    #
    v,v2,v3,v4=qx.staVars[0],qx.staVars[1],qx.staVars[2] ,qx.staVars[3]
    #
    
    df['vw_sum']=pd.rolling_sum(df[ksgn]*df['volume'],v);
    df['vw_vol']=df['volume'].rolling(window=v2,center=False).sum()
    df['vwap']=df['vw_sum']/df['vw_vol']
    #zt.prDF('df',df,22);print(df.describe())
    #ccc
    #
    #---sta.end.
    #trd_sub(qx,df,['vwap','>',v3, 'vwap','<',v4])
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[df.vwap>v3,'ktrd']=1
    df.loc[df.vwap<v4,'ktrd']=-1
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    return qx      
#========================
                                             
#----------xxxx 策略
def xxx_sta(qx):
    '''   
     xxx策略10
     xxx 指标，又称
    当 xxx>xxx,并且朝上，买入；
    当 xxx>xxx,并且朝下，卖出
    默认参数示例：
    qx.staVars=[20,75,115]    
 
'''    
    df=qx.wrkSybDat
    df=df.sort_index(ascending=True);
    #
    ksgn=qx.priceSgn
    v,v2,v3=qx.staVars[0],qx.staVars[1],qx.staVars[2] #,qx.staVars[3]
    #
    df=zta.STOD(df,v,ksgn);
    #df['stod1p']=df['stod'].shift(1)
    df['stok1p']=df['stok'].shift(1)
    #---sta.end.
    #
    df['kkd']=df['stok']/df['stod']*v2
    df['k1p']=df['stok']/df['stok1p']*v3
    #
    df['ktrd']=0  # =1:buy; =-1:sell
    df.loc[(df.kkd>100)&(df.k1p>100),'ktrd']=1
    df.loc[(df.kkd<100)&(df.k1p<100),'ktrd']=-1
    #zt.prDF('df',df,22);print(df.describe())
    #xxx2
    #
    trd_sub(qx,df,['ktrd','>',0, 'ktrd','<',0])
    #
    return qx    
