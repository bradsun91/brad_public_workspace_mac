import pandas as pd, numpy as np
from datetime import datetime
pd.set_option('max_colwidth',200)
# import yfinance as yf
import tushare as ts
ts.set_token('2f31c3932ead9fcc3830879132cc3ec8df3566550f711889d4a30f67')
import time, urllib
import glob
import os

# df = pro.index_daily(ts_code='399300.SZ', start_date='20180101', end_date='20181010')
data_folder = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/CH_database/"


def get_page(url):  #获取页面数据
    req=urllib.request.Request(url,headers={
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language':'zh-CN,zh;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
    })
    opener=urllib.request.urlopen(req)
    page=opener.read()
    return page

def get_index_history_byNetease(index_temp):
    """
    :param index_temp: for example, 'sh000001' 上证指数
    :return:
    """
    index_type=index_temp[0:2]
    index_id=index_temp[2:]
    if index_type=='sh':
        index_id='0'+index_id
    if index_type=="sz":
        index_id='1'+index_id
    url='http://quotes.money.163.com/service/chddata.html?code=%s&start=19900101&end=%s&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER;VATURNOVER'%(index_id,time.strftime("%Y%m%d"))

    page=get_page(url).decode('gb2312') #该段获取原始数据
    page=page.split('\r\n')
    col_info=page[0].split(',')   #各列的含义
    index_data=page[1:]     #真正的数据

    #为了与现有的数据库对应，这里我还修改了列名，大家不改也没关系
    col_info[col_info.index('日期')]='交易日期'   #该段更改列名称
    col_info[col_info.index('股票代码')]='指数代码'
    col_info[col_info.index('名称')]='指数名称'
    col_info[col_info.index('成交金额')]='成交额'

    index_data=[x.replace("'",'') for x in index_data]  #去掉指数编号前的“'”
    index_data=[x.split(',') for x in index_data]

    index_data=index_data[0:index_data.__len__()-1]   #最后一行为空，需要去掉
    pos1=col_info.index('涨跌幅')
    pos2=col_info.index('涨跌额')
    posclose=col_info.index('收盘价')
    index_data[index_data.__len__()-1][pos1]=0      #最下面行涨跌额和涨跌幅为None改为0
    index_data[index_data.__len__()-1][pos2]=0
    for i in range(0,index_data.__len__()-1):       #这两列中有些值莫名其妙为None 现在补全
        if index_data[i][pos2]=='None':
            index_data[i][pos2]=float(index_data[i][posclose])-float(index_data[i+1][posclose])
        if index_data[i][pos1]=='None':
            index_data[i][pos1]=(float(index_data[i][posclose])-float(index_data[i+1][posclose]))/float(index_data[i+1][posclose])

    # print(col_info)
    return [index_data,col_info]
# --------------------- 
# 版权声明：本文为CSDN博主「multiangle」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/u014595019/article/details/48445223

sh = get_index_history_byNetease("sh000001") 
df_ne = pd.DataFrame()
df_ne['columns_to_split'] = pd.DataFrame(sh).T[0]
col_list = list(pd.DataFrame(sh).T[1])[:12]
df_ne['columns_to_split'] = df_ne['columns_to_split'].apply(lambda x: str(x))
df_ne['columns_to_split'] = df_ne['columns_to_split'].apply(lambda x: x.replace("[", ""))
df_ne['columns_to_split'] = df_ne['columns_to_split'].apply(lambda x: x.replace("]", ""))

df_ne = df_ne['columns_to_split'].str.split(",", 12, expand = True)
df_ne.columns = col_list
df_ne.sort_values("交易日期", inplace = True)
df_ne.reset_index(inplace = True)
del df_ne['index']
df_ne.tail()



df_ne = df_ne[['交易日期', '开盘价', '最高价', '最低价', '收盘价', '成交量']]
df_ne = df_ne.iloc[1:, :]
df_ne['交易日期'] = df_ne['交易日期'].apply(lambda x: x[1:-1])
df_ne['收盘价'] = df_ne['收盘价'].apply(lambda x: x[2:-1])
df_ne['收盘价'] = df_ne['收盘价'] = df_ne['收盘价'].apply(lambda x: float(x))
df_ne['最高价'] = df_ne['最高价'].apply(lambda x: x[2:-1])
df_ne['最高价'] = df_ne['最高价'] = df_ne['最高价'].apply(lambda x: float(x))
df_ne['最低价'] = df_ne['最低价'].apply(lambda x: x[2:-1])
df_ne['最低价'] = df_ne['最低价'] = df_ne['最低价'].apply(lambda x: float(x))
df_ne['开盘价'] = df_ne['开盘价'].apply(lambda x: x[2:-1])
df_ne['开盘价'] = df_ne['开盘价'] = df_ne['开盘价'].apply(lambda x: float(x))
df_ne['成交量'] = df_ne['成交量'].apply(lambda x: x[2:-1])
df_ne['成交量'] = df_ne['成交量'] = df_ne['成交量'].apply(lambda x: float(x))
df_ne.columns = ['trade_date','open','high','low','close','volume']
df_ne['ts_code'] = 'sh000001'
df_ne.to_csv(data_folder+'sh000001.csv', index = False)

print("Downloading Process Finished!")



