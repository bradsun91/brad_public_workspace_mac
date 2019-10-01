# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime  

#import pandas.io.data as web
import pandas_datareader.data as web
import pandas_datareader as pdr
import fix_yahoo_finance as fy  


#zw.Quant
import zwSys as zw  
import zwQTBox as zwx

 
#-----------
fy.pdr_override()  
print(''' pandas_datareader模块的
      yahoo等金融数据接口变化很大
      请大家查询最新版本，或者参考
      其他数据接口的调用案例
      ''')

#===========================

'''

def zw_down_yahoo8code(qx):
    try:
        xcod=qx.code;
        #xdat= web.DataReader(xcod,"yahoo",start="1/1/1900");
        start=datetime.datetime(2017, 10, 1)  
        end=datetime.datetime(2017, 12, 31)
        #xdat=pdr.get_data_yahoo('AAPL',start="2017-01-01",end='2017-05-30')
        xdat=pdr.get_data_yahoo('AAPL',start,end)
        #xdat=pdr.get_data_yahoo('AAPL')
        #xdat=web.DataReader('AAPL',data_source='yahoo',start="1/1/2017");
        #xdat=web.DataReader('AAPL',data_source='google',start='3/14/2017',end='4/14/2017')
        fss=qx.rDay+xcod+".csv";print(fss);
        xdat.to_csv(fss);
    except IOError: 
        pass    #skip,error
    
        
#------------        
        

qx=zw.zwDatX(zw._rdatUS);
qx.prDat();

#
code='USA';qx.code=code;qx.rDay="tmp\\";
zw_down_yahoo8code(qx);
'''    