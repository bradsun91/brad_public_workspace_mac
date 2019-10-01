# -*- coding: utf-8 -*-

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
#df['unum9'], df['unum'], df['ucash']=zip(*df.apply(trd_sub010, args =([qx]+vlst_sub,),axis=1)) 

#from binance.client import Client

from kucoin.client import Client
#from idex.client import Client
#client = Client(address, private_key)

# get currencies
#----------------------

client = Client("", "")
c = client.get_currencies()
# get market depth
#depth = client.get_order_book('ETH_SAN')

#klines = client.get_historical_klines_tv("KCS-BTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")
            
#klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

print('ok',c)