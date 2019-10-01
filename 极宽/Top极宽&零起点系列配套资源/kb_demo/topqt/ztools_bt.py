# -*- coding: utf-8 -*- 
'''
TopQuant-简称TQ极宽智能量化回溯分析系统，培训课件-配套教学python程序

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2017.10.1 首发


网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  
  
文件名:ztools_bt.py
默认缩写：import ztools_bt as zbt
简介：Top极宽量化·回溯分析模块
 

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
#
import ztools as zt
import ztools_tq as ztq
import ztools_str as zstr
import ztools_sta as zsta
import ztools_msta as zmsta
import ztools_data as zdat
import ztools_draw as zdr

 
#-------------------
#   bt.init

   


def bt_init(qx):
    '''
    [输入参数]
        rdat，数据目录，一般日线数据是 rdatCN='/zDat/cn/day/'
        sybLst,股票池代码列表
        inxLst,指数池代码列表
        vlst，bt回溯初始化变量参数列表
            [btTim0Str='',btTim9Str='']
    '''
    #
    # pools_frd:sybLst & inxLst
    #ztq.tq_pools_init(qx)
    #print('\nwrk.inx,syb: ',qx.wrkInx,qx.wrkSyb)
    #
    #bt_init_tim(qx)
    #print('t0,t9:',qx.btTim0str,qx.btTim9str,qx.btTimNum)

    #
    qx.priceSgn,qx.trd_MulStaFlag,qx.trd_mode='close',False,1
    qx.trd_buyNum,qx.trd_buyMoney=1000,5000*2
    qx.trd_tim,qx.trd_NTimGap='',1
    qx.trd_sellNDiv0,qx.trd_sellNDiv,qx.trd_cnt=1,1,1
    #
    qx.usr_num9,qx.usrBuyMoney9,qx.usrMoney=0,0,qx.usrMoney0
    
    #
    #qx.trd_NTimGap=qx.staVars[-1]*0.1
    #
    return qx
           
   
#-------------bt.work






def bt_anz300(df9,sybLst,fgExt=0):
    print('\n bt_anz300')
    df9.index=pd.to_datetime(df9.index)
    df9.dropna(inplace=True)
    #1
    print('\n#1 rebase')
    df3=df9.rebase()
    df3.sort_index(inplace=True)
    #
    #2
    if fgExt>0:
        #2
        print('\n#2 fgExt')
        zt.prDF('\ndf2',df3)
        #3 
        print('\n#3 calc_stats')
        perf = df3.calc_stats()
        perf.display()
        perf.to_csv(path='tmp/perf01.csv')
        #
        if fgExt>1:
            #4
            xret=perf.display_lookback_returns()
            zt.prx('\n#4 @display_lookback_returns \n',xret)
            #
            
            #5.1
            print('\n#5.1 returns')
            ret=df3.to_returns().dropna()
            #5.2
            r_wts=ret.calc_mean_var_weights().as_format('.2%')
            zt.prx('\n#5.2 @mean_var_weights \n',r_wts)
            #5.3
            r_ercw=ret.calc_erc_weights().as_format('.2%')
            zt.prx('\n#5.3 @erc_weights \n',r_ercw)
            #5.4
            r_ivw=ret.calc_inv_vol_weights().as_format('.2%')
            zt.prx('\n#5.4 @inv_vol_weights \n',r_ivw)
            #
            #6
            print('\n#6 perf.sub.xxx')
            for syb9 in sybLst:
                print('\n@',syb9)
                x=perf[syb9]
                x.display_monthly_returns()
            #
            # 9  ----------------------
            print('\n#9 plot')
            
            #
            perf.plot(kind='line')
            plt.savefig('tmp/perf.png')
            
            perf.plot_correlation()
            plt.savefig('tmp/perf_corr.png')
            
            perf.plot_histograms()
            plt.savefig('tmp/perf_his.png')
            #
            perf.plot_scatter_matrix()
            plt.savefig('tmp/perf_scat.png')
    
    
    #
    return df3


def bt_main(qx,vsgn='vsum9',fgSyb9=False):
    print('\n@bt_main, @sta:',qx.staFun.__name__)
    df9=pd.DataFrame()
    mdn,mdn2=0,0
    k100={}
    fsta,vsta=qx.staFun,qx.staVars
    #
    for syb in qx.sybLst:
        qx=bt_init(qx)
        #
        qx.btTimNum,qx.rdatTyp,qx.wrkSyb=-1,'5m',syb
        qx.rdat=qx.rdat0+qx.rdatWeb+'/'
        #
        dat=zdat.df_frdXed(qx.rdat,syb,qx.priceSgn,qx.rdatTyp,qx.timFmtInv) #day:datType=''
        qx.wrkSybDat=zdat.df_kcut8tim(dat,'xtim',qx.btTim0str,qx.btTim9str)
        #qx.wrkSybDat['vcash9']=0
        #
        if not (vsgn in zsys.ohlcALst):
            print('@bt',syb)
            qx.staFun,qx.staVars=fsta,vsta
            #
            #print('\n@b0',fsta.__name__,vsta)
            qx.staFun(qx)
            #print('\n@b9',fsta.__name__,vsta)
            #
        #
        df=qx.wrkSybDat
        #uval9=df['vcash'].max()
        uval9=df['vcash9'].max()
        #df['vsum9']=uval9+df['vcash']
        df['vsum9']=df.vsum+(uval9-qx.usrMoney0)
        df9[syb]=df[vsgn]
        
        #
        df2=df[df.unum!=0]
        #print('dn2',len(df2.index))
        dn,dn2=len(df.index),len(df2.index)
        mdn,mdn2=mdn+dn,mdn2+dn2
        kdn=zt.xp100(dn2,dn,2)
        k100[syb+'_kdn'],k100[syb+'_dn2'],k100[syb+'_dn']=kdn,dn2,dn
        #
        #print(k100)
        #zt.prDF('df',df)
        df.to_csv('tmp/bt_'+syb+'.csv')
        #xx
        
    #
    #xxx
    #df9.to_csv('tmp/df9.csv')
    #zt.prDF('df9',df9)
    #xxx
    df3=bt_anz300(df9,qx.sybLst,False)
    #df3.to_csv('tmp/df9x.csv')
    #
    #df9.dropna(inplace=True)
    ravg=zdat.df_mean_last(df3)
    kmdn=zt.xp100(mdn2,mdn,2)
    ravg['kmdn'],ravg['mdn'],ravg['mdn2']=kmdn,mdn,mdn2
    #
    for syb in qx.sybLst:
        ravg[syb+'_kdn'],ravg[syb+'_dn2'],ravg[syb+'_dn']=k100[syb+'_kdn'],k100[syb+'_dn2'],k100[syb+'_dn']
    #print('\nravg');print(ravg)
    #xxx
    #
    
    #print('\n@ravg\n',ravg)
    
    #
    return df9,df3,ravg

#----bt.var


def bt_var010(qx,fsta,vsta,fgAnz=0):
    #
    qx.staFun,qx.staVars=fsta,vsta
    #qx.btTim0str,qx.btTim9str='','2018-02-22'
    #qx.btTim0str,qx.btTim9str='2018-01-01',''
    #qx.btTim0str,qx.btTim9str='2017-09-01',''
    #qx.btTim0str,qx.btTim9str='',''
    #
    #zbt.bt_main(qx,'avg') #week
    df9,_,ravg=bt_main(qx)
    #zt.prDF('df9',df9)
    #print(ravg)
    #xxx
    
    #
    if fgAnz>0:
        df3=bt_anz300(df9,qx.sybLst,fgAnz)
    #
    return ravg



def bt_var050(qx,fsta,vsta,fgAnz=2):
    #
    qx.staFun,qx.staVars=fsta,vsta
    #qx.btTim0str,qx.btTim9str='','2018-02-22'
    #qx.btTim0str,qx.btTim9str='2018-01-01',''
    #qx.btTim0str,qx.btTim9str='2017-09-01',''
    #qx.btTim0str,qx.btTim9str='',''
    #
    #zbt.bt_main(qx,'avg') #week
    df9,_,ravg=bt_main(qx)
    #zt.prDF('df9',df9)
    #print(ravg)
    #xxx
    
    #
    if fgAnz>0:
        df3=bt_anz300(df9,qx.sybLst,fgAnz)
    #
    return ravg

#---tst.mul.sta.xxx
    
#---tst.mul.sta.misc.xxx
    
def bt_mxlst010(qx):
    clst=qx.sybLst
    kdnlst=list(map(lambda x:x+'_kdn',clst))
    #dnlst=list(map(lambda x:x+'_dn',clst))
    dn2lst=list(map(lambda x:x+'_dn2',clst))
    cnams0=clst+kdnlst+dn2lst #+dnlst
    #
    vlst=['kmx','k','v','v2','v3','v4','kmdn','k100n','msta','sta','sta2']#,'mdn','mdn2']
    xlst=vlst+cnams0
    cnams=cnams0+['kmdn']#,'mdn','mdn2']
    #
    return cnams,cnams0,xlst,clst,vlst
    
def bt_mxVAnz(qx,rx,df9,ftg):
    cnams,cnams0,xlst,clst,vlst=bt_mxlst010(qx)
    ds=pd.Series(index=xlst)
    mfun,mvar=qx.staFun,qx.staVars
    [v,v2,v3,v4]=mvar
    #
    ds[cnams]=rx[cnams]
    ds.k,ds.v,ds.v2,ds.v3,ds.v4=rx.davg,v,v2,v3,v4
    ds.kmx=np.round(ds.k*rx.kmdn/100,2)
    #
    ds.msta=mfun.__name__
    ds.sta,ds.sta2='',''
    if len(qx.mstaPools)>1:
        [rsta1,rsta2]=qx.mstaPools
        ds.sta,ds.sta2=rsta1[0],rsta2[0]
    #
    k100cnt,k100n9=0,len(clst)
    #print('clst',clst)
    for ksgn in clst:
        dn=ds[ksgn]
        if dn>100:k100cnt+=1
    ds.k100n=zt.xp100(k100cnt,k100n9)
    #
    df9=df9.append(ds.T,ignore_index=True)
    if ftg!='':
        df9.to_csv(ftg,index=False)
    #
    df9v=pd.DataFrame(columns=vlst+clst)
    df9v=df9[vlst+clst]
    zt.prDF('\ndf9v',df9v,33)
    #
    return df9

#---tst.mul.sta.misc.tst.xxx


        
def bt_mx100(qx0,ftg='tmp/bt010.csv'):
    cnams,cnams0,xlst,clst,vlst=bt_mxlst010(qx0)
    #
    df9=pd.DataFrame(columns=xlst)
    #df9v=pd.DataFrame(columns=vlst+clst)
    #ds=pd.Series(index=xlst)
    #    
    fss='data/mx100.csv'
    mx10=pd.read_csv(fss,index_col=False)
    mx100=zdat.df2list(mx10)
    #zt.prDF('mx10',mx10)
    #
    #for xc,r in mx10.iterrows():
    for xc,r in enumerate(mx100):
        if xc>=0:
            print('\n',xc,'#',r[0])
            #
            qx=qx0
            zmsta.mx_sta4list(qx,r)
            #qx0.staFun,qx0.staVars=msta,mvar
            #
            _,_,rx=bt_main(qx)
            df9=bt_mxVAnz(qx,rx,df9,ftg)
            
           
