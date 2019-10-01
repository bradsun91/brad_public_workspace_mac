# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#import pandas.io.data as web
import pandas_datareader.data as web
import fix_yahoo_finance as fy  

#zw.Quant
import zwSys as zw  
import zwQTBox as zwBox


#--------
print(''' pandas_datareader模块的
      yahoo等金融数据接口变化很大
      请大家查询最新版本，或者参考
      其他数据接口的调用案例
      ''')

#==================
'''
fy.pdr_override()  


qx=zw.zwDatX(zw._rdatUS);

qx.code="AETI";fss="tmp\\"+qx.code+".csv";
zwBox.down_stk_yahoo010(qx,fss);

qx.code="EGAN";fss="tmp\\"+qx.code+".csv";
zwBox.down_stk_yahoo010(qx,fss)

qx.code="GLNG";fss="tmp\\"+qx.code+".csv";
zwBox.down_stk_yahoo010(qx,fss)

qx.code="SIMO";fss="tmp\\"+qx.code+".csv";
zwBox.down_stk_yahoo010(qx,fss)
'''