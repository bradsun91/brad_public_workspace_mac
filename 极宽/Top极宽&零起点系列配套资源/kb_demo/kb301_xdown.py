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
import time,arrow,os
import datetime
import pandas as pd
import numpy as np
#
import requests
import decimal
import hashlib
import json

import numpy as np
import pandas as pd

import pandas_datareader.data as web     # 网络财经数据接口库
import datetime as dt
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed
#
#
import zsys
import ztools as zt
import ztools_tq as ztq
import ztools_bt as zbt
import ztools_str as zstr
import ztools_web as zweb
import ztools_xcoin as zxc
import ztools_draw as zdr
import zpd_talib as zta
#----------------------------------


#1
jpy = web.DataReader('USDJPY','stooq')
jpy.to_csv('tmp/usdjpy2018.csv')
zt.prDF('#1 jpy',jpy)

#2
eur = web.DataReader('USDEUR','stooq')
eur.to_csv('tmp/USDEUR2018.csv')
zt.prDF('#2 eur',eur)

#3
btc = web.DataReader('BTCUSD','stooq')
btc.to_csv('tmp/BTCUSD2018.csv')
zt.prDF('#3 BTC',btc)


#-----------------------
print('ok')

