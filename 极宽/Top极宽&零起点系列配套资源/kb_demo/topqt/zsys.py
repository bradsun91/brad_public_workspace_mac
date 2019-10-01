# -*- coding: utf-8 -*- 
'''
TopQuant-简称TQ极宽智能量化回溯分析系统，培训课件-配套教学python程序

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2017.10.1 首发

网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  
  
文件名:z_sys.py
默认缩写：import zsys as zsys
简介：Top极宽量化·常用量化系统参数模块
 

'''
#

import sys,os,re
import arrow,bs4,random
import numexpr as ne  
#
import cpuinfo as cpu
import psutil as psu
from functools import wraps
#
import numpy as np
import pandas as pd
import tushare as ts
#import talib as ta

import matplotlib as mpl
import matplotlib.colors
from matplotlib import cm
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
#import multiprocessing
#

#


#

#import zpd_talib as zta

#-------------------
#----glbal var,const
__version__='2016.M10'

#------sys.war
cpu_num_core=8
cpu_num9=8
cpu_num=cpu_num9-1

tim0_sys=None
tim0_str=''

fn_time_nloop=5
fn_time_nloop5=500

#-----global.flag
web_get001txtFg=False  # @zt_web.web_get001txtFg

#-----bs4.findall
bs_get_ktag_kstr=''


#--colors
#10,prism,brg,dark2,hsv,jet
#10,,hot,Vega10,Vega20
cors_brg=cm.brg(np.linspace(0,1,10))
cors_hot=cm.hot(np.linspace(0,1,10))
cors_hsv=cm.hsv(np.linspace(0,1,10))
cors_jet=cm.jet(np.linspace(0,1,10))
cors_prism=cm.prism(np.linspace(0,1,10))
cors_Dark2=cm.Dark2(np.linspace(0,1,10))
#cors_Vega10=cm.Vega10(np.linspace(0,1,10))
#cors_Vega20=cm.Vega20(np.linspace(0,1,10))

#------str.xxx
sgnSP4='    '
sgnSP8=sgnSP4+sgnSP4

#-----FN.xxx
logFN=''

#--------------dir
raiLib='/aiLib/'
r_TDS=raiLib+'TDS/'
#-------------------
#---zDat.....
#
rdat0='/zDat/'
rdatCN0=rdat0+"cn/"
rdatCN=rdat0+"cn/day/"
rdatCNX=rdat0+"cn/xday/"
rdatInx=rdat0+"inx/"
rdatMin0=rdat0+"min/"
rdatTick=rdat0+"tick/"
rdatReal=rdat0+"real/"


#
#
#ohlc=['open','high','low','close']
#ohlc_date=['date']+ohlc
#
#---qxLib.xxxx
ohlcLst=['open','high','low','close']
ohlcVLst=ohlcLst+['volume']
ohlcVALst=ohlcLst+['volume','avg']
ohlcALst=ohlcLst+['avg']
ohlcDALst=ohlcLst+['avg','dprice']
#
ohlcDLst=['date']+ohlcLst
ohlcDVLst=['date']+ohlcLst+['volume']
ohlcExtLst=ohlcDLst+['volume','adj close']
#
ohlcTLst=['tim']+ohlcLst
ohlcTVLst=['tim']+ohlcLst+['volume']
#
#ohlcXTLst=['tim','xtim','xtimstamp']+ohlcLst
#ohlcXTVLst=['tim','xtim','xtimstamp']+ohlcLst+['volume']
ohlcXLst=['xstamp']+ohlcLst+['volume']
ohlcXTLst=['tim','utim','xstamp']+ohlcLst
ohlcXTVLst=['tim','utim','xstamp']+ohlcLst+['volume']
#
#xavg9Lst=['xavg1','xavg2','xavg3','xavg4','xavg5','xavg6','xavg7','xavg8','xavg9']
#xavg5Lst=['xavg1','xavg2','xavg3','xavg4','xavg5']
# 
#xhigh9Lst=['xhigh1','xhigh2','xhigh3','xhigh4','xhigh5','xhigh6','xhigh7','xhigh8','xhigh9']
#xhigh5Lst=['xhigh1','xhigh2','xhigh3','xhigh4','xhigh5']
# 
ma100Lst_var=[2,3,5,10,15,20,25,30,50,100]
ma100Lst=['ma_2','ma_3','ma_5','ma_10','ma_15','ma_20','ma_25','ma_30','ma_50','ma_100']
ma200Lst_var=[2,3,5,10,15,20,25,30,50,100,150,200]
#ma200Lst=['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20','ma_25', 'ma_30', 'ma_50', 'ma_100', 'ma_150', 'ma_200']
ma200Lst=['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20','ma_30', 'ma_50', 'ma_100', 'ma_150', 'ma_200']
#
ma030Lst_var=[2,3,5,10,15,20,25,30]
ma030Lst=['ma_2', 'ma_3', 'ma_5', 'ma_10', 'ma_15', 'ma_20', 'ma_25', 'ma_30']
#
priceLst=['price', 'price_next', 'price_change']
#
dateLst=['xyear','xmonth','xday','xday_week','xday_year','xweek_year']
timeLst=dateLst+['xhour','xminute']
#
TDS_xlst1=ohlcVALst
TDS_xlst2=ohlcVALst+ma100Lst
TDS_xlst9=TDS_xlst2+dateLst
#
#  keras.fun-lst.xxx
k_init_lst=['glorot_uniform','random_uniform','Zeros','Ones','Ones','RandomNormal','RandomUniform','TruncatedNormal','VarianceScaling','Orthogonal','Identiy','lecun_uniform','lecun_normal','glorot_normal','glorot_uniform','he_normal','he_uniform']
f_act_lst=[None,'elu','selu','relu','tanh','linear','sigmoid','softplus','hard_sigmoid'] #,'softsign,'
f_out_typ_lst=[None,'softmax']
#
f_opt_lst=['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
f_loss_lst=[ 'mse','mae','mape','msle','squared_hinge','hinge','binary_crossentropy','logcosh','kullback_leibler_divergence','poisson','cosine_proximity','categorical_hinge']
f_loss_typ_lst=['categorical_crossentropy'] #softmax
#


#open,high,low,close,volume
#,xtim,xyear,xmonth,xday,xday_week,xday_year,xweek_year,avg,ma_2,ma_3,ma_5,ma_10,ma_15,ma_20,ma_25,ma_30,ma_50,ma_100,
#
#xtrdName=['date','ID','mode','code','dprice','num','kprice','sum','cash'];
#xtrdNil=['','','','',0,0,0,0,0];
#qxLibName=['date','sybVal','cash','dret','val','downLow','downHigh','downDay','downKMax'];
#qxLibNil=['',0,0,0,0,0,0,0,0];  #xBars:DF
#  

f_inxNamTbl='xinx_name.csv'
#inxNamTbl=None  #全局变量，大盘指数的交易代码，名称对照表          
#inxLib={}    #全局变量，大盘指数，内存股票数据库
#inxLibCodX={}   #全局变量，大盘指数的交易代码等基本数据，内存股票数据库

#
f_sybNamTbl='xstk_name.csv'
#sybNamTbl=None  #全局变量，相关股票的交易代码，名称对照表          
#sybLib={}       #全局变量，相关股票的交易数据，内存股票数据库

#sybLibCodX={}   #全局变量，相关股票的交易代码等基本数据，内存股票数据库
#xlst=['code','name','ename','id','industry','id_industry','area','id_area']
#
  #100w
#
#qx_trdName=['ID','time','mode','code','dprice','num','sum','cash','syb-val','total'];
#qx_trdNil=['','','','',0,0,0,0,0,0];
#qx_trdName=['ID','time','cash','syb-val','total','pools'];
#qx_trdNil=['','',0,0,0,None];
#qx_trdName=['ID','time','cash','pools'];
#qx_trdNil=['','',0,None];
#qx_trdName=['ID','time','cash','code','num9','dnum','dprice','dsum'];
#qx_trdNil=['','',0,'',0,0,0,0];
#qx_trdName=['ID','time','cash','upools']
#qx_trdNil=['','',0,{}]

#
#qx_usrTrdName=['code','num','sum','dnum','dprice','dsum']; #'_name',{{xcod#1},{xcod#2}}
#usrsyb-dic：code:num; #'code','dprice','num','sum','mode',

#
#rdatUS=_rdat0+"us\\"

#-----pre.init
#mpl.style.use('seaborn-whitegrid');
pd.set_option('display.width', 450)    

#----------class.def

    
class TQ_bar(object):
    ''' 
    设置TopQuant项目的各个全局参数
    尽量做到all in one

    '''

    def __init__(self):  
        #----rss.dir
        #
        self.prjNam=''
        self.rdat0=rdat0
        self.rdat=rdat0
        self.rdatWeb=''
        
        #self.rinx=rdat0
        self.rdatTyp='5m' #5m,15m,30m,...,day,@pd
        #self.rdatFRQ='15T' # pd.resample, tq_xtim2FRQ
        #self.rdatWrk=''
        self.rtmp='tmp/'
        self.rtopo,self.topoSite,self.topoFtg0='','',''
        self.topoKSgn,self.topoSyb='avg','btcusd'
        self.topoNMax,self.topoKSize=100,500
        self.topoTim0str,self.topoTim9str='',''
        self.topoTim0,self.topoTim9='',''
        #
        #self.topoKlow0,self.topoKlow9=0,500
        #self.topoKavg0,self.topoKavg9=500,502
        #self.topoKup0,self.topoKup9=502,1000
        self.topoKlow0,self.topoKlow9=0,500
        self.topoKavg0,self.topoKavg9=500,510
        self.topoKup0,self.topoKup9=510,1000
        self.ktopo,self.ktopoGate=0,60   
        self.ktopoInx=100
        
        #
        self.ftg0=self.rtmp+self.prjNam
        
        #
        
        #
        #self.sybNamTbl=None
        #self.inxNamTbl=None
        #
        self.sybLst=[]   #inx=syb[0]
        self.sybNum=0
        self.sybPools={}
        self.mstaPools=[]
        #self.inxLst=[]
        #self.inxPools={}
        #
        
        
        #
        #self.tim0,self.tim9,self.tim_now=None,None,None
        #self.tim0Str,self.tim9Str,self.timStr_now='','',''
        #self.tim0wrk=arrow.now()
        #
        #wrk:working
        #self.wrkTim0,self.wrkTim9=arrow.now(),None
        #self.wrkTm0str,self.wrkTim9str=self.wrkTim0.format('YYYY-MM-DD HH:mm:ss'),''
        self.timFmt='YYYY-MM-DD'      #.format('YYYY-MM-DD HH:mm:ss')
        self.timFmtInv='%Y-%m-%d'   #.strftime('%Y-%m-%d %H:%M:%S')
        self.timDayFlag=True       #价格数据日线数据标志:=True，日线数据；=False，Min分时或者Tick数据
        self.timStrToday,self.timStrTodayExt=arrow.now().format('YYYY-MM-DD'),arrow.now().format('YYYY-MM-DD HH:mm:ss')
        self.wrkTim,self.wrkTimStr=None,''
        #
        self.wrkSyb,self.wrkSybDat,self.wrkSybInfo=None,None,None
        #self.wrkInx,self.wrkInxDat,self.wrkInxInfo=None,None,None
        #   avg,close
        self.wrkPrice,self.wrkPriceNext=0,0
        #self.wrkBar=None
        #
        self.priceSgn='avg'         #价格数据字段名称:avg;close，OHLC
        
 
        #---bt:backtest
        self.btTim0,self.btTim9,self.btTimNum=None,None,300
        self.btTim0str,self.btTim9str='',''
        #self.btTimLst=[]
        
        #
        self.staFun=None
        self.staVars=[]
        #
        #   bt,trade
        # m=0,init;m=1,num-Money;m=2,num-syb; m=3,ksize(0..1.0) ??
        self.trd_mode=1  
        
        #mode=1,2
        self.trd_sellNDiv0=1
        self.trd_sellNDiv=self.trd_sellNDiv0
        self.trd_buyNum,self.trd_buyMoney=1000,5000
        self.trd_NTimGap,self.trd_tim=0.5,''
        self.trd_cnt=1
        self.trd_MulStaFlag=False
        
        #
        #--usr.xxx
        self.usrMoney0=100*10000  #100w
        self.usrMoney,self.usrTotal=self.usrMoney0,self.usrMoney0   #100w
        self.usrBuyMoney9,self.usrBuyMoney0=0,0
        #
        self.usr_num9=0
        
           
        #
        self.aiModel={}   #dict
        self.aiMKeys=[]   #list
        #
        #--------------
        
#----------

#----------
        
   
if __name__ == "__main__":
    dn=psu.cpu_count(logical=False)
    print('main',dn)
    
    #initSysVar(True)
    
    