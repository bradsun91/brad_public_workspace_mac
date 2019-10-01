#coding=utf-8
# -*- coding: utf-8 -*- 
'''
Top极宽量化(原zw量化)，Python量化第一品牌 

网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  
    
TopQuant.vip ToolBox 2016
Top极宽·量化开源工具箱 系列软件 
by Top极宽·量化开源团队 2016.12.25 首发
  
文件名:ztools_data.py
默认缩写：import ztools_data as zdat
简介：Top极宽常用数据工具函数集
'''

import os,sys,io,re
import random,arrow,bs4
import numpy as np
import numexpr as ne
import pandas as pd
import tushare as ts

import requests
#
import cpuinfo as cpu
import psutil as psu
import inspect
#
import matplotlib as mpl
import matplotlib.colors
from matplotlib import cm


#
import zsys
import ztools as zt
import ztools_tq as ztq
import ztools_str as zstr
import ztools_web as zweb
import zpd_talib as zta

#

#-----------------------
'''

misc
#
df.xxx,pandas.xxx
    df.cov.
    df.get.
    df.cut.
#

'''

#-----------------------
#----------data.misc


#-----Series

    
def ds4x(x,inx=None,fgFlat=False):
    if fgFlat:
        x=x.flatten()[:]
    #
    ds=pd.Series(x)
    if len(inx)>0: ds.index=inx
    #
    return ds

def ds4lst(ds,ds0,vlst):
    for vss in vlst:
        ds[vss]=ds0[vss]
    #
    return ds

def ds4lst2x(ds,ds0,vlst,vlst0):
    for vss,vs0 in zip(vlst,vlst0):
        ds[vss]=ds0[vs0]
    #
    return ds


#----------df.misc
def df2list(df):
    d10= np.array(df)
    df2=d10.tolist()
    #
    return df2
    
#df['ktype']=np.round(df['price_change'])

#df['ktype'][df.ktype<900]=900
#df['close'][df.close>1100]=1100

def df2type20(df,ksgn='ktype',n9=10):
    #dsk
    #df['price_change']=df['price_next']/df['price']*100
    #
    df[ksgn]=np.round(df[ksgn])
    d0,d9=100-n9,100+n9
    #if df[ksgn]:
    df[ksgn][df[ksgn]<d0]=d0
    df[ksgn][df[ksgn]>d9]=d9
    df['ktype']=df['ktype'].astype(int)
    #
    return df


def df_xshift(df,ksgn='avg',num9=10):
    xsgn='x'+ksgn
    alst=[xsgn]
    df[xsgn]=df[ksgn].shift(-1)
    for xc in range(2,num9):
       xss=xsgn+'_'+str(xc)   
       df[xss]=df[ksgn].shift(-xc)
       alst.append(xss)
    #
    return df,alst

def df_mean_last22(df):
    d10=df.tail(1).values
    #print('d10',d10)
    d20=zt.lst_flatten02(d10)
    #print('d12',d12)
    #print('m',np.mean(d12))
    xd=np.mean(d20)
    #
    
    #
    return xd

def df_mean_last(df,nround=2):
    #c10=df.columns
    #ds=pd.Series(index=c10)
    ds=df.tail(1)
    #davg=ds.mean()
    davg=np.mean(ds.values)
    ds['davg']=davg
    #print('d10',d10)
    #d20=zt.lst_flatten02(d10)
    #print('d12',d12)
    #print('m',np.mean(d12))
    #xd=np.mean(d20)
    #
    rx=ds.ix[0]#.as_format('.2')
    #rx=zt.as_format(rx,'.2')
    rx=rx.round(nround)
    #
    return rx


def df_acc(df,ysgn0='avg2',ysgn='y_pred',k0=5):
    '''
    模型预测数据准确度分析，函数api接口是：
把输入的数据df，按原始数据和预测后的数据，进行准确度计算。
输入参数：
	df，输入数据集变量
	ysgn0，数据集当中的原始数据字段。
	y_pred，模型预测的数据集。
	k0，预测数据与实际数据的误差宽容度，默认为5%，如果两者误差小于5%，认为是正确数据。
返回参数：
	kacc，准确度，百分百形式。
	df，包含分析数据的输入数据扩展数据集。
	dfk，复合准确度要求的数据集。
'''
    
    df['diff']=abs(df[ysgn]-df[ysgn0])
    df['kdif']=round((df['diff']/df[ysgn0]*100),2)
    dfk=df[df['kdif']<k0]
    dn9,dnk=len(df.index),len(dfk.index)
    kacc=round(dnk/dn9*100,2)
    #
    print('\n@acc:{0:.2f}%, @kn:{1},dn:{2}\n'.format(kacc,dnk,dn9))
    #
    return kacc,df,dfk
    
#----------df2type
def df_type2float(df,xlst):
    for xsgn in xlst:
        df[xsgn]=df[xsgn].astype(float)

def df_type4mlst(df,nlst,flst):
    for xsgn in nlst:
        df[xsgn]=df[xsgn].astype(int)
        
    for xsgn in flst:
        df[xsgn]=df[xsgn].astype(float)
        
            
    
#----------df.xxx,pandas.xxx

#----------df.cov.xxx,pandas.xxx       
def df_2ds8xlst(df,ds,xlst):
    for xss in xlst:
        ds[xss]=df[xss]
    #
    
    #df9.to_csv(ftg,index=False,encoding='gbk')
    return ds

def df_2dic(df,xc):
    rx=df.head(xc).T.to_dict()
    return rx[xc]

def df_2ds(df,xc):
    rx=df.head(xc).T.to_dict();#print('\nrx',rx.keys())
    klst=list(rx.keys());#print('@k',klst)
    kss=klst[xc-1];#print('@k',kss)
    r2=pd.Series(rx[kss]);#print('\nr2',r2)
    return r2
    
    
#----------df.get.xxx,pandas.xxx    
    
def df_get1k(df,v1,k1):
    df2=df[df[v1]==k1]
    return df2
          
def df_get4lst(df,vlst,klst):
    df2=df
    for k,v in zip(klst,vlst):
        df2=df_get1k(df2,v,k)
        #df2[df2[k]==v]
    #
    return df2
    
def df_get8kvar(df,vsgn,klst,vlst):
    df2,dat=df,None
    for k,v in zip(klst,vlst):
        #print('k,v,',k,v)
        if len(df2.index)>0:
            df2=df2[df2[k]==v]
        #print('k,v,',k,v)
    #
    #rint('df2,',df2)
    if len(df2.index)>0:
        dat=df2[vsgn].values[0]
        #print('v,d',vsgn,dat)
    #
    return dat
    
def df_get8kvar01(df,vsgn,vlst):
    df2,dat=df,None
    df9=pd.DataFrame(columns=df.columns)
    #ds=pd.Series(index=clst,dtype=str)
    for xc, rx in df.iterrows():
        vss=rx[vsgn]
        if vss in vlst:
            df9=df9.append(rx.T,ignore_index=True)
    
        #df2=df2[df2[k]==v]
        #print('k,v,',k,v)
    #
    return df9
    
        
    
def df_get8tim(df,ksgn,kpre,kn9,kpos):
    #@ zdr.dr_df_get8tim
    #
    xdf=pd.DataFrame(columns=['nam','dnum'])
    ds=pd.Series(['',0],index=['nam','dnum'])
    for xc in range(1,kn9+1):
        xss,kss='{0:02d}'.format(xc),'{0}{1:02d}'.format(kpre,xc)
        df2=df[df[ksgn].str.find(kss)==kpos]
        ds['nam'],ds['dnum']=xss,len(df2['gid'])
        xdf=xdf.append(ds.T,ignore_index=True)
        #print(xc,'#',xss,kss)
    #
    xdf.index=xdf['nam']
    return xdf


 
#----------
def df_get_tim2x(df,timsgn):
    df=df.sort_values(timsgn)
    tim0Str,tim9tr=df[timsgn].values[0],df[timsgn].values[-1]
    #
    return tim0Str,tim9tr   
    

#----------df.cut.xxx,pandas.xxx            
def df_kcut8tim(df,ksgn,tim0str,tim9str):
    '''
    把输入的数据df，按时间参数，进行切割。
输入参数：
	df，输入数据变量
	ksgn，数据切割使用的字段名称，默认为xtim。ksgn为空字符串时，使用index索引字段比较时间。
	tim0str，数据切割起始时间，空字符串不进行切割。
	tim9str，数据切割结束时间，空字符串不进行切割。
返回参数：
	df，切割后的数据集，pandass的DataFrame表格格式。

    '''
    if ksgn=='':
        if tim0str!='':df2=df[df.index>=tim0str]
        else:df2=df
        if tim9str!='':df3=df2[df2.index<=tim9str]
        else:df3=df2
    else:
        if tim0str!='':df2=df[df[ksgn]>=tim0str]
        else:df2=df
        if tim9str!='':df3=df2[df2[ksgn]<=tim9str]
        else:df3=df2
    #
    return df3
        
def df_kcut8yearlst(df,ksgn,ftg0,yearlst):
    for ystr in yearlst:
        tim0str,tim9str=ystr+'-01-01',ystr+'-12-31'
        df2=df_kcut8tim(df,ksgn,tim0str,tim9str)
        ftg=ftg0+ystr+'.dat';print(ftg)
        df2.to_csv(ftg,index=False,encoding='gb18030')
    
def df_kcut8myearlst(df,ksgn,tim0str,ftg0,yearlst):
    for ystr in yearlst:
        tim9str=ystr+'-12-31'
        df2=df_kcut8tim(df,ksgn,tim0str,tim9str)
        ftg=ftg0+ystr+'.dat';print(ftg)
        df2.to_csv(ftg,index=False,encoding='gb18030')
    
#----------df.xed
def df_xappend(df,df0,ksgn,num_round=3,vlst=zsys.ohlcDVLst):
    if (len(df0)>0):   
        df2 =df0.append(df)     
        df2=df2.sort_values([ksgn],ascending=True);
        df2.drop_duplicates(subset=ksgn, keep='last', inplace=True);
        #xd2.index=pd.to_datetime(xd2.index);xd=xd2
        df=df2
        
    #
    df=df.sort_values([ksgn],ascending=False);
    df=np.round(df,num_round);
    df2=df[vlst]
    #
    return df2
    
#----------df.xtim.xxx

def df_xtim2mtim(df,ksgn='xtim',fgDate=False):    
    df['xyear']=df[ksgn].apply(zstr.str_2xtim,ksgn='y')
    df['xmonth']=df[ksgn].apply(zstr.str_2xtim,ksgn='m')
    df['xday']=df[ksgn].apply(zstr.str_2xtim,ksgn='d')
    #
    df['xday_week']=df[ksgn].apply(zstr.str_2xtim,ksgn='dw')
    df['xday_year']=df[ksgn].apply(zstr.str_2xtim,ksgn='dy')
    #df['xday_month']=df['xtim'].apply(zstr.str_2xtim,ksgn='dm')
    df['xweek_year']=df['xtim'].apply(zstr.str_2xtim,ksgn='wy')
    #
    df['xhour']=df[ksgn].apply(zstr.str_2xtim,ksgn='h')
    df['xminute']=df[ksgn].apply(zstr.str_2xtim,ksgn='t')
    
    #
    if fgDate:
        df=df.drop(['xhour','xminute'],axis=1)
    #
    return df

#----------df.xdat.ed.xxx
def df_xed_nextDay(df,ksgn='avg',newSgn='xavg',nday=10):    
    #df['avg']=df[zsys.ohlcLst].mean(axis=1).round(2)
    for i in range(1,nday):
        xss=newSgn+str(i)
        df[xss]=df[ksgn].shift(-i)
    #
    return df
    
        
def df_xed_ailib(df,ksgn='avg',fgDate=True):
    #   xed.avg
    df=df.sort_index(ascending=True);
    if ksgn=='avg':
        df[ksgn]=df[zsys.ohlcLst].mean(axis=1)
    else:
        df[ksgn]=df[ksgn]
    #   xed.time
    df['xtim']=df.index
    df=df_xtim2mtim(df,'xtim',fgDate)
    #   xed.ma.xxx
    df=zta.mul_talib(zta.MA,df, ksgn,zsys.ma100Lst_var)
    #
    #   xed.xavg.xxx,predict,y_data
    df=df_xed_nextDay(df,ksgn,'x'+ksgn,10)
    #
    df=df.round(2)
    df=df.dropna()
    #
    return df
    

def df_xed_xtyp(df,kmod='3',k0=99.5,k9=100.5,sgnTyp='ktype',sgnPrice='price_change'):
    kmod=kmod.lower()
    if kmod=='n':
        df[sgnTyp]=df[sgnPrice].apply(zt.iff2ntype,v0=k0,v9=k9)                 #v0=95,v9=110)
    elif kmod=='3':   
        df[sgnTyp]=df[sgnPrice].apply(zt.iff3type,d0=k0,d9=k9,v3=3,v2=2,v1=1)   #k0=99.5,k9=100.5):
    else:    
        df[sgnTyp]=df[sgnPrice].apply(zt.iff2type,d0=k0,v1=1,v0=0)              #100.5
    #    
    df['y']=df[sgnTyp].astype(float)
    ydat=df['y'].values
    return df,ydat

#df_test['ktype']=df_test['price_change'].apply(zt.iff3type,d0=99.5,d9=100.5,v3=3,v2=2,v1=1)
    
def df_xed_xtyp2x(df_train,df_test,kmod='3',k0=99.5,k9=100.5,sgnTyp='ktype',sgnPrice='price_change'):
    df_train,y_train=df_xed_xtyp(df_train,kmod,k0,k9,sgnTyp,sgnPrice)
    df_test,y_test=df_xed_xtyp(df_test,kmod,k0,k9,sgnTyp,sgnPrice)
    #
    return df_train,df_test,y_train,y_test

#----------df.x_xxx    
    

def df_xget(fss,tim0str='',tim9str='',ktrain=0.7):
    '''
    根据文件名读取数据，并根据时间参数切割数据，最终把数据按ktrain设置的比例，分为train训练数据集tst测试数据集。
输入参数：
	fss，数据文件名
	tim0str，tim9str，数据切割起始结束时间，为空时不处理。
	ktrain，数据切割比例，默认总数据的70%为train训练数据
返回参数：
	df_train，训练数据集，pandas的DataFrame格式。
	df_test，测试数据集，pandas的DataFrame格式。

    '''
    df=pd.read_csv(fss,index_col=0)
    #
    df['xtim']=df.index
    df.drop_duplicates('xtim', keep='last', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    #
    df['avg']=(df['open']+df['high']+df['low']+df['close'])/4
    df['avg2']=df['avg'].shift(-1)
    df.dropna(inplace=True)
    #
    c10=set(df.columns)
    if ('vol' in c10)and(not ('volume' in c10)):
        df['volume']=df['vol']
        df.drop(['vol'],inplace=True,axis=1)
    if 'cap' in c10:df.drop(['cap'],inplace=True,axis=1)
    #
    df=df_cut8tim(df,'xtim',tim0str,tim9str)
    #
    x10=list(df.index)
    dn9,xn9=len(df.index),len(x10)
    ntrain,ntst=int(dn9*ktrain),int(dn9*(1-ktrain))-1
    print('\n@dn9,xn9:',dn9,xn9,'@ntrain,ntst:',ntrain,ntst)
    #------------
    df_train,df_tst=df.head(ntrain),df.tail(ntst)
    #
    return df_train,df_tst

#----------df.file
def df_rdcsv_tim0(fss,ksgn,tim0):
    xd0= pd.read_csv(fss,index_col=False,encoding='gbk') 
    #print('\nxd0\n',xd0.head())
    if (len(xd0)>0): 
        #xd0=xd0.sort_index(ascending=False);
        #xd0=xd0.sort_values(['date'],ascending=False);
        xd0=xd0.sort_values([ksgn],ascending=True);
        #print('\nxd0\n',xd0)
        xc=xd0.index[-1];###
        _xt=xd0[ksgn][xc];#xc=xd0.index[-1];###
        s2=str(_xt);
        #print('\nxc,',xc,_xt,'s2,',s2)
        if s2!='nan':
            tim0=s2.split(" ")[0]        
            
    #
    return xd0,tim0        


#-------------------pools
def df_initXed(df,ksgn='avg',datTyp='15m',timFMT='%Y-%m-%d %H:%M:%S'):
    df.index=pd.to_datetime(df.index)
    df.index.rename('tim_inx',inplace=True)
    #
    clst=set(df.columns);
    if 'utim' in clst:df=df.drop(['utim'],axis=1)
    #if 'tim' in clst:df.rename(index=str, columns={"tim": "time"})
    df['xtim']=df.index.strftime(timFMT) 
    df['avg']=df[zsys.ohlcLst].mean(axis=1)
    df['dprice']=df[ksgn]
    #print(df.tail())
    #x=df['xtim'][0]
    #print(x,type(x))
    #print(type(df.index))
    #
    df=df.drop_duplicates(['xtim'])
    df=df.sort_index(ascending=True);
    #
    xsgn=ztq.tq_xtim2FRQ(datTyp) #???
    df=df.resample(xsgn, closed='left').pad()
    df.dropna(inplace=True)
    #
    df=df.sort_index()
    return df
    
def df_xedBTC1kDiv(df):
    #zt.prDF('df',df)
    df[zsys.ohlcLst]=df[zsys.ohlcLst]/1000
    #zt.prDF('df',df)
    return df

def df_xedBTC1kMul(df):
    #zt.prDF('df',df)
    df[zsys.ohlcLst]=df[zsys.ohlcLst]*1000
    #zt.prDF('df',df)
    return df

def df_frdXed(rss,syb,ksgn='avg',datTyp='15m',timFMT='%Y-%m-%d %H:%M:%S'):
    if datTyp=='':fcod=rss+syb+'.csv'
    else:fcod=rss+syb+'_'+datTyp+'.csv'
    #
    df=pd.read_csv(fcod,index_col=0)
    #
    ksyb=syb.lower().find('btc')
    if ksyb==0:df=df_xedBTC1kDiv(df)
    if ksyb>1:df=df_xedBTC1kMul(df)
    #    
    df=df_initXed(df,ksgn,datTyp,timFMT)
    #
    
    #
    return df
    
    
def pools_frd(rss,clst,ksgn='avg',datTyp='15m',timFMT='%Y-%m-%d %H:%M:%S'):
    print('\nclst:',clst)
    dats={}
    i,n9=0,len(clst)
    for syb in clst:
        df=df_frdXed(rss,syb,ksgn,datTyp,timFMT)
        i+=1
        print(i,'/',n9,syb)
        #
        dats[syb]=df
    #
    return dats

def pools_link010(dat,pools,clst,ksgn='avg',inxFlag=False):
    i,n9,inxSgn=0,len(clst),''
    if inxFlag:inxSgn='x'
    
    for xcod in clst:
        i+=1
        print(i,'/',n9,xcod)
        df=pools[xcod]
        #
        #if ksgn=='avg':df['avg']=df[zsys.ohlcLst].mean(axis=1)
        #
        #print(df.tail())
        dat[inxSgn+xcod]=df[ksgn]
    #
    dat=dat.round(9)
    return dat

def pools_link2x(stkPools,clst,inxPools,xlst,ksgn):
    dat=pd.DataFrame()
    #
    dat=pools_link010(dat,stkPools,clst,ksgn)
    dat=pools_link010(dat,inxPools,xlst,ksgn,True)
    #
    return dat
    

def pools_link2qx(qx,ksgn,fgInx=True):
    dat=pd.DataFrame()
    #
    
    if fgInx:
        #dat=pools_link010(dat,qx.inxPools,qx.inxCodeLst,ksgn,True)
        dat=pools_link010(dat,qx.inxPools,qx.inxLst,ksgn)
    #
    dat=pools_link010(dat,qx.sybPools,qx.sybLst,ksgn)
    
    #qx.wrkPriceDat=dat
    return dat



#-------------------file

def f_links8codes(rss,clst):
    i,n9=0,len(clst)
    df9=pd.DataFrame()
    for cod in clst:
        fss=rss+cod+'.csv'
        i+=1
        print(i,'/',n9,fss)
        #
        df=pd.read_csv(fss)
        df=df[zsys.ohlcDVLst]
        df9=df9.append(df)
    #
    return df9


def f_links_TDS(rss,clst,ksgn='avg',fgDate=True):
    i,n9=0,len(clst)
    df9=pd.DataFrame()
    for cod in clst:
        fss=rss+cod+'.csv'
        i+=1
        print(i,'/',n9,fss)
        #
        df=pd.read_csv(fss)
        df=df_xed_ailib(df,ksgn,fgDate)
        df9=df9.append(df)
    #
    return df9

#-------------------file.df.rw
def fdf_wr(df,ftg,kn=9,fgFlt=False):
    if ftg!='':
        #
        if fgFlt:
            df=df.drop_duplicates()
            df=df.dropna()
        #        
        ftmp='tmp/'+zstr.str_fnRnd('.csv')
        kss='%.'+str(kn)+'g'
        df.to_csv(ftmp,index=False,float_format=kss)
        df=pd.read_csv(ftmp,index_col=False)
        df.to_csv(ftg,index=False,float_format=kss)    
        os.remove(ftmp) 
    
def fdf_lnk_wr(df9,df,ksgn,ftg,kn=9,fgInv=False):  
    if len(df.index)>0:
        df9=df9.append(df,ignore_index=True)
        #df9=df9.drop_duplicates()
        df9=df9.drop_duplicates(subset=ksgn, keep='last')
        df9=df9.dropna()
        if ksgn!='':
            if fgInv:df9=df9.sort_values(ksgn,ascending =False)
            else:df9=df9.sort_values(ksgn)
    
        fdf_wr(df9,ftg,kn)
    #
    return df9
    
            #    
#-------------------file.aidat

def f_rd_xdat(fdat,xlst,ysgn='y'):
    '''
        no ysgn:'y' in xlst
    '''
    df=pd.read_csv(fdat,index_col=False)
    df['y']=df[ysgn].astype(float)
    xdat,ydat=df[xlst].values,df['y'].values
    #
    mlst=xlst+['y']
    if ysgn!='y':mlst=mlst+[ysgn]
    
    df2=df[mlst]
    #
    return df2,xdat,ydat

#-------------------file.TDS
def frd_TDS_sub(fdat,ksgn,xlst,fgChange=False):
    df=pd.read_csv(fdat)
    df['price_next']=df[zsys.xavg9Lst].max(axis=1)
    df['price'],df['y']=df[ksgn],df['price_next']
    df['price_change']=df['price_next']/df['price']*100
    if fgChange:df['y']=df['price_change']
    #
    xdat,ydat=df[xlst].values,df['y'].values
    clst,mlst=[ksgn,'y','price','price_next','price_change'],xlst
    for css in clst:
        if not (css in mlst):mlst=mlst+[css]
    #    
    df2=df[mlst]
    #
    return df2,xdat,ydat
    
def frd_TDS(rdat,fsgn,ksgn,xlst,fgChange=False):
    #rss='/ailib/TDS/'
    f_train,f_test=rdat+fsgn+'_train.csv',rdat+fsgn+'_test.csv'
    df_train,x_train,y_train=frd_TDS_sub(f_train,ksgn,xlst,fgChange)
    df_test,x_test, y_test  =frd_TDS_sub(f_test,ksgn,xlst,fgChange)
    #
    return df_train,df_test ,x_train,y_train,x_test, y_test
