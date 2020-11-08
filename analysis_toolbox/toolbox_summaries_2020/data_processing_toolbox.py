import psycopg2
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyecharts
import urllib3,time,csv,datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import matplotlib.dates as mpd
import plotly.plotly as py
import plotly.offline as py_offline
import plotly.graph_objs as go
from IPython.display import clear_output

%matplotlib inline



# ===================================================================================================================
# 11-08-2020
# 简单高效地运用pivottable来转换数据格式，用以进行corr analysis或者根据date来plot returns

fund_nav_df
"""
fund_nav_df长什么样子：

	ts_code	ann_date	end_date	unit_nav	accum_nav	accum_div	net_asset	total_netasset	adj_nav	update_flag
0	511990.SH	20201107	20201106	1.0000	None	None	NaN	NaN	12735.5844	1
1	511990.SH	20201106	20201105	1.0000	None	None	NaN	NaN	12734.8939	1
2	511990.SH	20201105	20201104	1.0000	None	None	NaN	NaN	12734.1857	1
3	511990.SH	20201104	20201103	1.0000	None	None	NaN	NaN	12733.4646	1
4	511990.SH	20201103	20201102	1.0000	None	None	NaN	NaN	12732.7244	1
...	...	...	...	...	...	...	...	...	...	...
168	515030.SH	20200304	20200303	0.9819	0.9819	None	NaN	NaN	0.9819	0
169	515030.SH	20200229	20200228	0.9494	0.9494	None	NaN	NaN	0.9494	0
170	515030.SH	20200228	20200226	0.9944	0.9944	None	1.070217e+10	1.07022e+10	0.9944	0
171	515030.SH	20200222	20200221	1.0007	1.0007	None	NaN	NaN	1.0007	1
172	515030.SH	20200221	20200220	1.0000	1	None	1.076288e+10	1.07629e+10	1.0000	1

其特征就是，所有的tickers信息都被上下concat在一起，而不是根据date的columns被merge到一起，因此很不方便进行相关性分析或者plot其序列在同一张图里

解决方法：

直接用Pivot table来转换数据格式

"""

transformed_df = fund_nav_df_test_2.pivot_table(index='ann_date', columns=['ts_code'], values='adj_nav')

"""
transformed_df 长什么样子？
ts_code	511990.SH	515030.SH
ann_date		
20121229	10001.1902	NaN
20130101	10004.1876	NaN
20130105	10008.9664	NaN
20130112	10010.6887	NaN
20130119	10011.8450	NaN
...	...	...
20201103	12732.3685	1.2921
20201104	12733.4646	1.2889
20201105	12734.1857	1.3169
20201106	12734.8939	1.3890
20201107	12735.5844	1.3823

"""


# ===================================================================================================================
# 05-17-2019 
# 分割一栏into两栏split one column into two columns
# new data frame with split value columns 
new = data["Name"].str.split(" ", n = 1, expand = True) 
  
# making separate first name column from new data frame 
data["First Name"]= new[0] 
  
# making separate last name column from new data frame 
data["Last Name"]= new[1] 


# ===================================================================================================================
# 4-9-2019 updated
# 计算程序运行时间
import datetime
starttime = datetime.datetime.now()
print ("Executing...")
endtime = datetime.datetime.now()
duration = (endtime - starttime).seconds
print ("Execution takes {} seconds".format(duration))



# ===================================================================================================================
# 4-2-2019 updated
# Python - 利用zip函数将两个列表(list)组成字典(dict)
keys = ['a', 'b', 'c']
values = [1, 2, 3]
dictionary = dict(zip(keys, values))
print (dictionary)
# 输出:
# {'a': 1, 'c': 3, 'b': 2}

# ===================================================================================================================
# 3-24-2019
# download data from yahoo finance and plot correlation heatmaps on downloaded stocks' returns
def corr_heatmaps_yf(symbol_list, price_col, start_str, end_str, corr_thresh):
    """
    Documentation: 
    1. start/end_str is of the format of, e.g. "2017-09-15"
    2. corr_thresh ranges from -1 to 1
    
    """
    df = yf.download(symbol_list, start = start_str, end = end_str)
    stacked = df.stack().reset_index()
    stacked.columns = ['date', 'tickers', 'close', 'close2', 'high', 'low', 'open','volume']
    stacked_ = stacked[['date', 'tickers', 'open', 'high', 'low', 'close', 'volume']]
    stacked_col = stacked_[['date', 'tickers', price_col]]
    stacked_col_pvt = pd.pivot_table(stacked_col, values = price_col, index = 'date', columns = 'tickers')
    stacked_col_pvt_pctchg = stacked_col_pvt.pct_change()
    fig, ax = plt.subplots(figsize = (12, 8))
    sns.heatmap(stacked_col_pvt_pctchg.corr()[(stacked_col_pvt_pctchg.corr()>corr_thresh)|(stacked_col_pvt_pctchg.corr()<-corr_thresh)], ax = ax, cmap = 'Blues', vmax = 1.0, vmin = -1.0, annot=True)
    plt.xlabel('stocks', fontsize = 15)
    plt.ylabel('stocks', fontsize = 15)
    plt.xticks(fontsize = 17)
    plt.yticks(fontsize = 17)
    return stacked_col_pvt_pctchg


# ===================================================================================================================
# 3-7-2019 updated
# 最终目标：plot correlation heatmaps
# 重要新知识点：利用reduce一次性merge多个csv dataframes
# 下载上证50个个股数据：
import tushare as ts
from functools import reduce #重要知识点：reduce

sz50 = ts.get_sz50s()
sz50_code_list = list(sz50['code'])
folder_all = "C:/Users/workspace/brad_public_workspace_on_win/non_code_files_brad_public_workspace_on_win/brad_public_workspace_on_win_non_code_files/SH_tongliang/data/SZ50_daily_data/1998_2019_all_51/"
n = 0
for code in sz50_code_list[48:]:
    cons = ts.get_apis()
    df = ts.bar(code, conn=cons, freq='D', start_date='1998-01-01', end_date='2019-03-06')
    df.reset_index(inplace=True)
    df = df[['datetime', 'open', 'high', 'low', 'close', 'vol', 'amount']]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount']
    # 看有多少可以被下载下来的文件：
    len_ = len(df)
    n = n+1
    df.to_csv(folder_all+code+"_1998_2019.csv", index = False)
    print ("No.{}, {}的数据量：{}，起始时间: {}".format(n, code, len_, df['date'].values[-1]))


stock_list = []
len_ = 0
for fname in glob.glob(all_csvs)[:]:
#     print (fname)
    stock = pd.read_csv(fname)
    stock = stock.sort_values('date')
    stock = stock[['date','close']]
    stock['pct_chg'] = stock['close'].pct_change()
    ticker = fname[-20:-14]
    stock.columns = ['date', 'close', ticker]
    stock = stock[['date', ticker]].dropna()
    stock['date'] = pd.to_datetime(stock['date'])
#     stock.set_index('date', inplace=True)
    stock_list.append(stock)
    # print ("Length of {}: {}".format(ticker, len(stock)))
    # print (stock.head(20))
    # len_ = len_+len(stock)
    # print ("Total length:{}".format(len_))
    # print ("===========")

# 先位置后使用reduce铺路：创造一个merge的函数：
def merge_df(df1, df2):
    df1.sort_values('date', inplace = True)
    merged = df1.merge(df2, on = 'date', how = 'outer')
    merged.sort_values('date', inplace = True)
    return merged

# 重要知识点：reduce
merged_all = reduce(merge_df, stock_list)
merged_all.set_index('date', inplace=True)

# 最后一步：plot heatmap:
fig, ax = plt.subplots(figsize = (40, 30))
sns.heatmap(merged_all.corr()[abs(merged_all.corr())>-2], ax = ax, cmap = 'Blues', vmax = 1.0, vmin = -1.0, annot=True)
plt.xlabel('stocks', fontsize = 15)
plt.ylabel('stocks', fontsize = 15)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17);


# ===================================================================================================================
# 2-25-2019 udpated
print("{:.2f}".format(number)) # 两位小数


# ===================================================================================================================
# 2-21-2019 udpated
# 生成信号代码：小于0为-1；大于0为1；等于0为0：
signal_df['signal'] = signal_df['signal'].apply(lambda x: 1 if x>0 else -1 if x < 0 else 0)

# 根据信号dataframe，生成回测收益结果：

def calc_single_performance(signal_df, price_col):
    """
    1. date是经过函数pd.to_datetime()处理过后的index；
    2. signal的值为0或者-1或者1，分别代表不持仓、空头信号和多头信号；
    3. price可以是close, open等需要当作计算收益基础的价格数据；
    
    signal_df的格式示例如下：
    =============================
                  price   signal
       date
    2017-07-28     256.3    -1
    2017-07-29     259.5     0
    =============================
    """
    signal_df['price_diff'] = signal_df[price_col].diff()
    signal_df['forward_signal'] = signal_df['signal'].shift(1)
    signal_df['returns'] = signal_df['forward_signal']*signal_df['price_diff']
    signal_df['cum_returns'] = signal_df['returns'].cumsum()
    return signal_df



# ===================================================================================================================
# 12_13_2018: 使用tushare
import tushare as ts
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

ts.set_token("2f31c3932ead9fcc3830879132cc3ec8df3566550f711889d4a30f67")
pro = ts.pro_api()




# ===================================================================================================================
# 12_3_2018: 数据全部标准化到百位，除了BTC

import pandas as pd, numpy as np
from datetime import datetime
import psycopg2

def all_assts_from_sql(asst1, asst2, asst3, asst4, asst5, asst6, asst7, asst8, sql_limit_num, location, till_date):
    conn = psycopg2.connect(database="bitmexdata", user="postgres", password="tongKen123", host="128.199.97.202",
                            port="5432")
    asset1 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst1,
                                                                                                    sql_limit_num)
    asset2 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst2,
                                                                                                    sql_limit_num)
    asset3 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst3,
                                                                                                    sql_limit_num)
    asset4 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst4,
                                                                                                    sql_limit_num)
    asset5 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst5,
                                                                                                    sql_limit_num)
    asset6 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst6,
                                                                                                    sql_limit_num)
    asset7 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst7,
                                                                                                    sql_limit_num)
    asset8 = "SELECT time, open, high, low, close, volume from {} order by id desc limit {}".format(asst8,
                                                                                                    sql_limit_num)
    df1 = pd.read_sql(asset1, con=conn)
    df2 = pd.read_sql(asset2, con=conn)
    df3 = pd.read_sql(asset3, con=conn)
    df4 = pd.read_sql(asset4, con=conn)
    df5 = pd.read_sql(asset5, con=conn)
    df6 = pd.read_sql(asset6, con=conn)
    df7 = pd.read_sql(asset7, con=conn)
    df8 = pd.read_sql(asset8, con=conn)
    conn.close()
    #     return df1, df2, df3, df4, df5, df6, df7, df8
    df1.to_csv(location + "{}".format(till_date) + asst1 + ".csv", index=False)
    df2.to_csv(location + "{}".format(till_date) + asst2 + ".csv", index=False)
    df3.to_csv(location + "{}".format(till_date) + asst3 + ".csv", index=False)
    df4.to_csv(location + "{}".format(till_date) + asst4 + ".csv", index=False)
    df5.to_csv(location + "{}".format(till_date) + asst5 + ".csv", index=False)
    df6.to_csv(location + "{}".format(till_date) + asst6 + ".csv", index=False)
    df7.to_csv(location + "{}".format(till_date) + asst7 + ".csv", index=False)
    df8.to_csv(location + "{}".format(till_date) + asst8 + ".csv", index=False)


# standardize and reset the price:
# file_suffix example: 'z18_1d.csv'
# added_note example: 'u18z18乘'
# till_date example: '11_10_'
def reset_price(location, till_date, file_suffix, added_note):
    symlist = ['ada', 'bch', 'eth', 'eos', 'trx', 'xrp', 'ltc']

    #     for s in symlist:
    #         if s == 'ada':
    #             c = 10000000
    #         elif s == 'bch':
    #             c = 1000
    #         elif s == 'eos':
    #             c = 100000
    #         elif s == 'eth' or s == 'ltc':
    #             c = 10000
    #         elif s == 'trx':
    #             c = 100000000
    #         elif s == 'xrp':
    #             c = 10000000
    for s in symlist:
        if s == 'ada':
            c = 100000000  #
        elif s == 'bch':
            c = 10000  #
        elif s == 'eos':
            c = 1000000  #
        elif s == 'eth':
            c = 10000
        elif s == 'ltc':
            c = 100000  #
        elif s == 'trx':
            c = 100000000
        elif s == 'xrp':
            c = 10000000

        # b = pd.read_csv(location + till_date + s + file_suffix,header=None)
        b = pd.read_csv(location + "{}".format(till_date) + s + file_suffix + ".csv")
        b.iloc[:, [1, 2, 3, 4]] = b.iloc[:, [1, 2, 3, 4]].astype(float)
        b.iloc[:, [1, 2, 3, 4]] = (b.iloc[:, [1, 2, 3, 4]]) * c
        b.to_csv(location + 'res_' + till_date + s + file_suffix + '.csv',
                 sep=',', header=False, index=False, float_format='%.4f')


location = "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/extracting_and_transforming_data/"
till_date = "12_03_og_all_hundreds_"
frequency = "_1d"
file_suffix = 'u18z18' + frequency
added_note = 'u18z18'

all_assts_from_sql("ada" + file_suffix, "bch" + file_suffix, "eos" + file_suffix,
                   "eth" + file_suffix, "ltc" + file_suffix, "trx" + file_suffix,
                   "xbtusd" + frequency, "xrp" + file_suffix, 1000,
                   "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/extracting_and_transforming_data/",
                   till_date)

reset_price(location, till_date, file_suffix, added_note)

btc_df = pd.read_csv(location + till_date + "xbtusd" + frequency + ".csv", engine="python", header=None)
btc_df = btc_df.iloc[1:, :]
btc_df.to_csv(location + "res_" + till_date + "xbtusd" + frequency + ".csv",
              sep=',', header=False, index=False, float_format='%.4f')


# ===================================================================================================================
# 11_24_1min 数据变成四个小时：

location = "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/11_24_bitmex分钟线_multiplied/bitmex分钟线1124/"

def chg_col(df):
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    return df

def time_to_timestamp(timestr):
    """
    时间字符串转unix时间戳
    :param str: 时间字符串
    :return: unix时间戳，str类型
    """
    dt = datetime.strptime(str(timestr), '%Y-%m-%d %H:%M:%S')
    timestamp = time.mktime(dt.timetuple())
    return str(int(timestamp))

def myresample(df, period, min):
    convrted_df = df.resample(period).last()
    convrted_df['open'] = df['open'].resample(period).first()
    convrted_df['high'] = df['high'].resample(period).max()
    convrted_df['low'] = df['low'].resample(period).min()
    convrted_df['close'] = df['close'].resample(period).last()
    convrted_df['volume'] = df['volume'].resample(period).sum()
    # Keep rows with at least 5 non-NaN values
    convrted_df.dropna(thresh=5, inplace=True)
    convrted_df.index = convrted_df['time']
    convrted_df['time'] = pd.DatetimeIndex(time_translation(t, min) for t in convrted_df['time'])
    convrted_df['timestamp'] = [time_to_timestamp(i) for i in convrted_df['time']]
    return convrted_df

def time_translation(ltime, min):
    res_time = (datetime.datetime.strptime(ltime, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=min)).strftime(
        '%Y-%m-%d %H:%M:%S')
    return res_time


# Create a fucntion that transforms 1 min of data into 4-hour data:
def transform_1min_into_4hr(csv_file, location):
    df = pd.read_csv(location+csv_file, engine="python", header=None)
    df.index = pd.to_datetime(df.iloc[:, 0])
    df = chg_col(df)
    resampled_df = myresample(df, "4h", -59)
    resampled_df.index = resampled_df['time']
    del resampled_df['time']
    resampled_df.reset_index(inplace = True)
    resampled_df = resampled_df.iloc[:-1, :]
    return resampled_df

resampled_ada = transform_1min_into_4hr("res_adau18z18乘10000000.csv", location)
resampled_trx = transform_1min_into_4hr("res_trxu18z18乘100000000.csv", location)
resampled_bch = transform_1min_into_4hr("res_bchu18z18乘1000.csv", location)
resampled_eos = transform_1min_into_4hr("res_eosu18z18乘100000.csv", location)
resampled_eth = transform_1min_into_4hr("res_ethu18z18乘10000.csv", location)
resampled_ltc = transform_1min_into_4hr("res_ltcu18z18乘10000.csv", location)
resampled_xrp = transform_1min_into_4hr("res_xrpu18z18乘10000000.csv", location)
resampled_btc = transform_1min_into_4hr("xbtusd_1m.csv", location)



# ===================================================================================================================
# 11_12_2018:
# Example: 
# rank_loc = "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/"
# rank_file = "11_12_ranks_all.csv"

def clean_rank_data(rank_loc, rank_file):
    rank = pd.read_csv(rank_loc+rank_file, header=None)
    rank.columns = ['Ranks']
    rank['Dates'] = np.nan
    for i, date in enumerate(rank['Ranks']):
        if "-" in rank['Ranks'][i]:
            rank['Dates'][i] = rank['Ranks'][i]
    rank.ffill(inplace=True)
    rank_ = rank[rank['Ranks'] != rank['Dates']]
    rank_['Dates'] = rank_['Dates'].apply(lambda x: x.replace("排名", ""))
    rank_['Assets'], rank_['Rank'] = rank_['Ranks'].str.split(' ', 1).str
    rank_['Rank'] = rank_['Rank'].apply(lambda x: int(x))
    rank_cleaned = pd.pivot_table(rank_, values='Rank', columns='Assets', index = 'Dates')
    return rank_cleaned


# ===================================================================================================================
# 转变不同周期、用Plotly实时plot，并且以蜡烛图的形式，加上交易信号
# 转变数据周期：T/5T/15T/30T/H/2H/4H/D/W/M
def resample(df, period):
    convrted_df = df.resample(period).last()
    convrted_df['open'] = df['open'].resample(period).first()
    convrted_df['high'] = df['high'].resample(period).max()
    convrted_df['low'] = df['low'].resample(period).min()
    convrted_df['close'] = df['close'].resample(period).last()
    convrted_df['volume'] = df['volume'].resample(period).sum()
    # Keep rows with at least 5 non-NaN values
    convrted_df.dropna(thresh=5, inplace=True)
    convrted_df.index = convrted_df['time']
    convrted_df['time'] = pd.DatetimeIndex(convrted_df['time'])
    return convrted_df

# 为能够plot蜡烛图做timestamps处理的准备：
def cnvrt_date(convrted_df):
    cnvrted_date_df = convrted_df.copy()
    cnvrted_date_df['date'] = mpd.date2num(cnvrted_date_df['time'].dt.to_pydatetime())
    return cnvrted_date_df

# 计算两个资产的相对差累计值：
def two_assets_tmsum(cnvrted_date_df1, cnvrted_date_df2, N4):
    cnvrted_date_df1['close_shifted'] = cnvrted_date_df1['close'].shift(1)
    cnvrted_date_df2['close_shifted'] = cnvrted_date_df2['close'].shift(1)
    T1 = cnvrted_date_df1['close'].diff()/cnvrted_date_df1['close_shifted']
    T2 = cnvrted_date_df2['close'].diff()/cnvrted_date_df2['close_shifted']
    TM = T1*(1-N4/100) - T2*(N4/100)
    tmsum_sr = TM.cumsum()
    return tmsum_sr
    
# 计算相对差的移动平均快慢线：
def MAs_of_tmsum(tmsum_sr, N1, N2):
    MA1 = tmsum_sr.ewm(span= N1).mean() #快线
    MA2 = tmsum_sr.ewm(span= N2).mean() #慢线   
    return MA1, MA2
    
# Plot TMSUM图线：
def plot_tmsum(tmsum_sr):
    py_offline.init_notebook_mode()
    tmsum_df = pd.DataFrame(tmsum_sr, columns=['tmsum'])
    tmsum_df = go.Scatter(x = tmsum_df.index,
                          y = tmsum_df['tmsum'])
    data = [tmsum_df]
    return py_offline.iplot(data, filename='TMSUM')
    
def plot_tmsum_MAs(MA1, MA2):
    # Here I didn't use offline's version of plotly, going forward  
    # will need to be consistent when moving to pycharm for plotting
    ma1_df = pd.DataFrame(MA1, columns=['MA1'], index = MA1.index)
    ma2_df = pd.DataFrame(MA2, columns=['MA2'], index = MA2.index)
    trace1 = go.Scatter(x = ma1_df.index, 
              y = ma1_df['MA1'])
    trace2 = go.Scatter(x = ma2_df.index, 
              y = ma2_df['MA2'])
    data = [trace1, trace2]
    fig = go.Figure(data=data, 
    #                     layout=layout
                   )
    return py.iplot(fig, filename='plot_tmsum_MAs')
    
    
# Plot 资产实时图
def plot_candlestick(cnvrted_date_df):
    py_offline.init_notebook_mode()
    candle_df = go.Candlestick(x = cnvrted_date_df.index,
                               open = cnvrted_date_df['open'],
                               high = cnvrted_date_df['high'],
                               low  = cnvrted_date_df['low'],
                               close = cnvrted_date_df['close'])
    data = [candle_df]
    return py_offline.iplot(data, filename='Candle Stick', image_width=2, image_height=4)

# Build a function that plots charts of two moving averages with their crossover trading signals 
# Here we want to make sure that both MA1_sr and MA2_sr are: series with timestanmps as their indexes

def MA_crossover_plot_signals(MA1_sr, MA2_sr):
    # 构建一个由 MA1和 MA2构成的dataframe：
    ma_signal_df = pd.DataFrame(MA1_sr, columns=['MA1'], index= MA1_sr.index)
    ma_signal_df['MA2'] = MA2_sr
    # 用两者的差表示 MA1在位置上高于还是低于MA2，负值说明低于，正值说明高于：
    ma_signal_df['MA1_mns_MA2'] = ma_signal_df['MA1'] - ma_signal_df['MA2']
    # 将正负值同义转换为二元的 1或者 -1便于观察和处理
    ma_signal_df['signs'] = ma_signal_df['MA1_mns_MA2']*abs(1/(ma_signal_df['MA1_mns_MA2']))
    # 用当前值和前一个周期的值决定当前状态是金叉信号还是死叉信号
    ma_signal_df['pre_signs'] = ma_signal_df['signs'].shift(1)
    ma_signal_df['signals'] = ma_signal_df['signs'] - ma_signal_df['pre_signs']
    ma_signal_df['signals_alert'] = ma_signal_df['signals'].apply(lambda x: "金叉" if x==2 else "死叉" if x==-2 else "无信号")
    # 将所有出现信号的rows挑出来建立一个字典：
    ma_df_with_signals = ma_signal_df[ma_signal_df['signals_alert'] != "无信号"]['signals_alert']
    signal_dict = dict(ma_df_with_signals)
    # Add each and all signal information(dictionary format) to annotations(list format) 
    # so that it can be put into the go.Layout() function.
    annotations = []
    each_dict = {}
    for i, k in enumerate(signal_dict):
        each_dict['x'] = k
        each_dict['y'] = ma_signal_df['MA1'].loc[k]
        each_dict['text']=ma_signal_df['signals_alert'].loc[k]
        each_dict['showarrow']=True
        each_dict['arrowhead']=7
        each_dict['ax']=0
        each_dict['ay']=-40

        annotations.append(each_dict.copy())
    # 用plotly进行绘图，包括了之前处理好的annotations，作为显示信号的功能
    ma1_df = pd.DataFrame(tmsum_ma1, columns=['MA1'], index = tmsum_ma1.index)
    ma2_df = pd.DataFrame(tmsum_ma2, columns=['MA2'], index = tmsum_ma2.index)
    trace1 = go.Scatter(x = ma1_df.index, 
              y = ma1_df['MA1'])
    trace2 = go.Scatter(x = ma2_df.index, 
              y = ma2_df['MA2'])
    layout = go.Layout(
        showlegend=False,
        annotations=annotations
    )
    data = [trace1, trace2]
    fig = go.Figure(data=data, 
                    layout=layout)
    return py.iplot(fig, filename='plot_MAs_with_signals')
    

def print_correction_rate(asst, rank_df, close_res_df):
    rank_asst = rank[['Date',asst]]
    rank_asst.index = rank_asst['Date']
    del rank_asst['Date']
    rank_asst.index = pd.to_datetime(rank_asst.index)
    rank_asst['date'] = rank_asst.index.astype(str)
    close_res_df['date'] = close_res_df.index.astype(str)
    merged = rank_asst.merge(close_res_df, on = 'date')
    merged['next_day_pct'] = merged['pct_chg']
    merged.dropna(inplace=True)
    print ("数字货币： ", merged.columns[0])
    print ("检验时间段：",merged['date'].values[0]," to ",merged['date'].values[-1])
    # correct prediction rate:
    try:
        correct_long_prdct = len(merged[((merged[asst] == 7) | (merged[asst] == 6) | (merged[asst] == 5)) &(merged['pct_chg']>0)])/len(merged[((merged[asst] == 7) | (merged[asst] == 6) | (merged[asst] == 5))])
    except ZeroDivisionError:
        print (print ('没有过做多信号'))
    else:
        print ("预测准确率：多头",correct_long_prdct)
        
    try:    
        correct_short_prdct = len(merged[((merged[asst] == 0) | (merged[asst] == 1) | (merged[asst] == 2)) &(merged['pct_chg']<0)])/len(merged[((merged[asst] == 0) | (merged[asst] == 1) | (merged[asst] == 2))])    
    except ZeroDivisionError:
        print (print ('没有过做空信号'))
    else:      
        print ("预测准确率：空头",correct_short_prdct)























