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
# 统计中的排列组合：
def fac(n):
    factorial = 1
    if n == 0:
         factorial = 0
    else:
        for i in range(1, n + 1):
            factorial *= i
#             print(factorial)
    return factorial

def C_m_n(m, n):
    combo_num = fac(m)/(fac(m-n)*fac(n))
    return combo_num
    # 从m里挑选n个
    
def A_m_n(m, n):
    combo_num = fac(m)/fac(n)
    return combo_num

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
# 2-1-2019 udpated
# 随机森林单品种回测商品期货，AIO函数——研究参数random_state，参数优化一条龙：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.style
plt.style.use("ggplot")
%matplotlib inline

import sys
sys.version


location =  "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/commodities_data/12_28_commodities_daily/"
file = "j9000_d.csv"
exported_signal_file = "j9000_d_testing_signals.csv"
n = 10
test_size = 1/6


def preprocess_df(location, file):
    df = pd.read_csv(location+file, engine="python", header=None)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interests']
    return df

def get_indicators(data, n, indicator):
    
    ###### Step 1: Calculate necessary time series ######
    up, dw = data['close'].diff(), -data['close'].diff()
    up[up<0], dw[dw<0] = 0, 0
    # default set to be 12-period ema as the fast line, 26 as the slow line:
    macd = data['close'].ewm(12).mean() - data['close'].ewm(26).mean()
    # default set to be 9-period ema of the macd
    macd_signal = macd.ewm(9).mean()
    
    ###### Step 2: Create dataframe and fill with technical indicators: ######
    indicators = pd.DataFrame(data=0, index=data.index,
                              columns=['sma', 'ema', 'momentum', 'rsi', 'macd'])
#     indicators['date'] = data['date']
    indicators['sma'] = data['close'].rolling(n).mean()
    indicators['ema'] = data['close'].ewm(n).mean()
    indicators['momentum'] = data['close'] - data['close'].shift(n)
    indicators['rsi'] = 100 - 100 / (1 + up.rolling(n).mean() / dw.rolling(n).mean())
    indicators['macd'] = macd - macd_signal
    indicators.index = data['date']
    return indicators[[indicator]]

def get_data(df, n):
    # technical indicators
    sma = get_indicators(df, n, 'sma')
    ema = get_indicators(df, n, 'ema')
    momentum = get_indicators(df, n, 'momentum')
    rsi = get_indicators(df, n, 'rsi')
    macd = get_indicators(df, n, 'macd')
    tech_ind = pd.concat([sma, ema, momentum, rsi, macd], axis = 1)
    df.index = df['date']
    close = df['close']
    direction = (close > close.shift()).astype(int)
    target = direction.shift(-1).fillna(0).astype(int)
    target.name = 'target'
    master_df = pd.concat([tech_ind, close, target], axis=1)
    return master_df

def rebalance(unbalanced_data, rblnc_rs):
    # Sampling should always be done on train dataset: https://datascience.stackexchange.com/questions/32818/train-test-split-of-unbalanced-dataset-classification
    # Separate majority and minority classes
    if unbalanced_data.target.value_counts()[0]>unbalanced_data.target.value_counts()[1]:
        print ("majority:0, length: {}; minority:1, length: {}".format(unbalanced_data.target.value_counts()[0],unbalanced_data.target.value_counts()[1]))
        data_minority = unbalanced_data[unbalanced_data.target==1] 
        data_majority = unbalanced_data[unbalanced_data.target==0] 
    else:
        print ("majority:1, length: {}; minority:0, length: {}".format(unbalanced_data.target.value_counts()[1],unbalanced_data.target.value_counts()[0]))
        data_minority = unbalanced_data[unbalanced_data.target==0] 
        data_majority = unbalanced_data[unbalanced_data.target==1] 
    # Upsample minority class
    n_samples = len(data_majority)
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=rblnc_rs)
    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])
    data_upsampled.sort_index(inplace=True)
    # Display new class counts
    data_upsampled.target.value_counts()
    return data_upsampled

def normalize(x):
    scaler = StandardScaler()
    # 公式为：(X-mean)/std  计算时对每个属性/每列分别进行。
    # 将数据按期属性（按列进行）减去其均值，并除以其方差。得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。
    x_norm = scaler.fit_transform(x.values)
    x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)
    return x_norm

def scores(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    
    
def train_test_validate(master_df, train_start, train_end, test_size, tts_rs, rblnc_rs, plot=True): 
    # train_start example: '2011-01-01'
    # test_size defaults as 1/6, 
    # test_size: parameter
    
    data = master_df.copy()
    data.index = pd.to_datetime(data.index)
    if plot == True:
        print ("Plotting data's close price series")
        ax = data[['close']].plot(figsize=(20, 5))
        ax.set_ylabel("Price (￥)")
        ax.set_xlabel("Time")
        plt.show()
    else:
        pass
    data_train = data[train_start : train_end]
    # Sampling should always be done on train dataset: https://datascience.stackexchange.com/questions/32818/train-test-split-of-unbalanced-dataset-classification
    data_train = rebalance(data_train, rblnc_rs).dropna()
    # y as the label target 
    y = data_train.target
    # X as the dataframe with their values to be normalized
    X = data_train.drop('target', axis=1)
    X = normalize(X)
    
    data_val = data[train_end:]
    data_val.dropna(inplace=True)
    # y_val as the label target in the validation period
    y_val = data_val.target
    # X_val as the dataframe with their values to be normalized in the validation period
    X_val = data_val.drop('target', axis=1)
    # normalize X_val dataframe
    X_val = normalize(X_val)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = tts_rs)
    print ("-----------------------------------------------")
    print ("X length: ", len(X))
    print ("X_val length: ", len(X_val))
    print ("X_train length: ", len(X_train))
    print ("X_test length: ", len(X_test))
    print ("-----------------------------------------------")
    print ("y length: ", len(y))
    print ("y_val length: ", len(y_val))
    print ("y_train length:", len(y_train))
    print ("y_test length:", len(y_test))
    print ("-----------------------------------------------")
    # Outputs of this function are 8 variables from above.
    return data, X, X_val, X_train, X_test, y, y_val, y_train, y_test
    
    
def optimize_model_paras(X_train, y_train, X_test, y_test):
    # first take a look at the default model's results:
    model = RandomForestClassifier(random_state=5)
    print ("Training default model...")
    model.fit(X_train, y_train)
    print ("Default model's scores:")
    scores(model, X_test, y_test)
    # set up parameters to be optimized
    grid_data =   {'n_estimators': [10, 50, 100],
                   'criterion': ['gini', 'entropy'],
                   'max_depth': [None, 10, 50, 100],
                   'min_samples_split': [2, 5, 10],
                   'random_state': [1]}
    grid = GridSearchCV(model, grid_data, scoring='f1').fit(X_train, y_train)
    print ("-----------------------------------------------")
    print ("Model's best parameters: ")
    print(grid.best_params_)
    model = grid.best_estimator_
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    print("Performance of the train_test datasets: ")
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    print ("----------------------------------------------------")
    print ("Optimized Model from the train_test dataset: ", model)
    
    # Validate optimized model:
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    print ("----------------------------------------------------")
    
    optimized_model = model
    return optimized_model

def train_test_backtest(optimized_model, X, y, X_train, y_train):
    rf_model = optimized_model
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    print("train_test datasets performance: ")
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    print ("----------------------------------------------------")
    mask = y_pred.copy()
    np.place(mask, y_pred==0, -1)
    mask = np.roll(mask, 1)
    data_returns = data['close'].diff()
    data_returns = data_returns[X.index]
    model_returns = mask * data_returns
    model_cum = model_returns.cumsum()
    equity = model_returns.sum()
    start_close = data["close"][X.index[0]]
    performance = equity / start_close * 100
#     ax = model_returns.plot(figsize=(15, 8))
#     ax.set_ylabel("Returns (￥)")
#     ax.set_xlabel("Time")
#     plt.show()
    ax = model_cum.plot(figsize=(15, 8))
    ax.set_ylabel("Cummulative returns (￥)")
    ax.set_xlabel("Time")
    plt.show()
    return model_cum, equity, performance, mask, y_pred, data_returns


# Trading system: testing real performance:
def validate_backtest(optimized_model, X_val, y_val, X_train, y_train):
    rf_model = optimized_model
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    print("validation datasets performance: ")
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))
    print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    print ("----------------------------------------------------")
    mask = y_pred.copy()
    np.place(mask, y_pred==0, -1)
    mask = np.roll(mask, 1)
    data_returns = data['close'].diff()
    data_returns = data_returns[X_val.index]
    model_returns = mask * data_returns
    model_cum = model_returns.cumsum()
    equity = model_returns.sum()
    start_close = data["close"][X_val.index[0]]
    performance = equity / start_close * 100
#     ax = model_returns.plot(figsize=(15, 8))
#     ax.set_ylabel("Returns ($)")
#     ax.set_xlabel("Time")
#     plt.show()
    ax = model_cum.plot(figsize=(15, 8))
    ax.set_ylabel("Cummulative returns ($)")
    ax.set_xlabel("Time")
    plt.show()
#     print (pd.DataFrame(model_cum)) # 对了
    return model_cum, equity, performance, mask, y_pred, data_returns

# Create signal file that is to be imported to TB:
def create_TB_signal_df(df, X_val, y_pred, y_val, mask, data_returns, exported_file):
    print ("Processing signal dataframe...")
    master_pred_df = X_val.copy()
    master_pred_df['y_pred'] = y_pred
    master_pred_df['y_val'] = y_val
    master_pred_df['mask'] = mask
    master_pred_df['data_returns'] = data_returns
    master_pred_df['model_returns'] = mask * data_returns
    master_pred_df_dt = master_pred_df.copy()
    master_pred_df_dt.reset_index(inplace = True)
    
    print ("Processing original OHLCV dataframe...")
    df_dt = df.copy()
    del df_dt['date']
    df_dt.reset_index(inplace= True)
    df_dt['date'] = pd.to_datetime(df_dt['date'])

    print ("Merging signal dataframe and OHLCV dataframe...")
    master_pred_df_dt = master_pred_df_dt[['date', 'mask']]
    merged = df_dt[['date', 'open', 'high', 'low', 'close']].merge(master_pred_df_dt, on = 'date')
    merged.columns = ['date', 'open', 'high', 'low', 'close', 'signals']
    
    print ("Exporting final signal file...")
#     merged.to_csv(location + exported_file, index = False, header = False)
    print ("All done!")
    
    return merged, master_pred_df



rblnc_rs = [1,5]
tts_rs = [1,5]
RFC_rs = [1,5]

backtest_records = {
                    'rblnc_rs':[],
                    'tts_rs':[],
                    'RFC_rs':[],
                    'cum_returns':[]}

def RF_rs_loop_AIO(rblnc_rs, tts_rs, RFC_rs):

    # Part 1:
    df = preprocess_df(location, file)

    # Part 2:
    master_df = get_data(df, n)

    # Part 3:
    data, X, X_val, X_train, X_test, y, y_val, y_train, y_test = train_test_validate(master_df, '2011-01-01','2017-01-01', test_size, tts_rs, rblnc_rs, False)

    # Part 4: if we already have all optimized parameters we just run this step: 
    optmzd_model = RandomForestClassifier(
                bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=RFC_rs, verbose=0, warm_start=False
                )

    # Part 5: See in-sample backtest
    model_cum, equity, performance, mask, y_pred, data_returns = train_test_backtest(optmzd_model, X, y, X_train, y_train)

    # Part 6: See out-of-sample backtest
    model_cum_, equity_, performance_, mask_, y_pred_, data_returns_ = validate_backtest(optmzd_model, X_val, y_val, X_train, y_train)
#     print (pd.DataFrame(model_cum_))  # 已解决
    return model_cum_
    


backtest_curves = pd.DataFrame([])
for rs_1 in rblnc_rs:
    for rs_2 in tts_rs:
        for rs_3 in RFC_rs:
            model_cum_ = RF_rs_loop_AIO(rs_1, rs_2, rs_3)
            print ("rblnc_rs: ", rs_1)
            print ("tts_rs: ", rs_2)
            print ("RFC_rs: ", rs_3)
#             backtest_records['rblnc_rs'].append(rs_1)
#             backtest_records['tts_rs'].append(rs_2)
#             backtest_records['RFC_rs'].append(rs_3)
            print ("=============================================All Finished.==================================================")
            print ("model_cum_: ", pd.DataFrame(model_cum_).head(3))
            backtest_curves = pd.concat([backtest_curves, pd.DataFrame(model_cum_)], axis=1)

backtest_curves.plot(figsize=(16, 8))



# ===================================================================================================================
# 12_17_2018_计算最大未创新高时长
def AIO_get_max_down_dur(root, file):
    # delete #Bar column in the file before putting root and file into the function.
    df = pd.read_csv(root+file, engine="python")
    
    df.columns = ['time', 'long_margin', 'short_margin', 'capital_available', 'floating_equity', 'trading_costs', 'static_equity', 'accum_returns']
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)

    equity = df.copy()
    equity.index = equity['time']
    equity = equity[['floating_equity']]
    
    """权益最长未创新高的持续时间"""
    logger.debug('---equity in analysis---: {}'.format(equity))
    max_equity = 0
    duration = pd.Timedelta(0)  # 时间间隔为 0
    date_list = []
    date_dur = pd.DataFrame(columns=['duration', 'start', 'end'])
    for i in range(equity.shape[0]):
        if max_equity <= equity.values[i][0]: #
            max_equity = equity.values[i][0]
            date_list.append(equity.index[i])
    logger.debug('---date_list---: {}'.format(date_list))
    for j in range(len(date_list) - 1):  # len()-1
        duration_ = date_list[j + 1] - date_list[j]

        date_dur = date_dur.append(pd.Series(
            [duration_, date_list[j], date_list[j + 1]], index=['duration', 'start', 'end']), ignore_index=True)

    date_dur = date_dur.sort_values('duration')
    start_date = date_dur.iloc[-1]['start']
    if equity.iloc[-1].values <= max_equity:
        deltta = equity.index[-1] - date_list[-1]
        start_date = date_list[-1]
        end_date = equity.index[-1]
    else:
        end_date = date_dur.iloc[-1]['end']
    date = start_date.strftime('%Y-%m-%d') + ' - ' + \
        end_date.strftime('%Y-%m-%d')
    logger.debug('---date in analysis---: {}'.format(date))
#     return date
    return date_dur, equity


# ===================================================================================================================
# 12_13_2018: 使用tushare
import tushare as ts
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

ts.set_token("2f31c3932ead9fcc3830879132cc3ec8df3566550f711889d4a30f67")
pro = ts.pro_api()


# ===================================================================================================================
# 12_12_2018: 计算未创新高的天数以及对应的日期，并导出排名


def duration_of_equity_not_reaching_high(equity):
    """权益最长未创新高的持续时间"""
    # equity = context.fill.equity.df
    logger.debug('---equity in analysis---: {}'.format(equity))
    max_equity = 0
    duration = pd.Timedelta(0)  # 时间间隔为 0
    date_list = []
    date_dur = pd.DataFrame(columns=['duration', 'start', 'end'])
    for i in range(equity.shape[0]):
        if max_equity <= equity.values[i][0]: #
            max_equity = equity.values[i][0]
            date_list.append(equity.index[i])
    logger.debug('---date_list---: {}'.format(date_list))
    for j in range(len(date_list) - 1):  # len()-1
        duration_ = date_list[j + 1] - date_list[j]

        date_dur = date_dur.append(pd.Series(
            [duration_, date_list[j], date_list[j + 1]], index=['duration', 'start', 'end']), ignore_index=True)
        #
        # if duration < duration_:
        #     duration = duration_
        #     date_dict[duration] = [date_list[i], date_list[i + 1]]

    # date = date_dict[max(date_dict)][0] + '-' + date_dict[max(date_dict)][1]
    date_dur = date_dur.sort_values('duration')
    start_date = date_dur.iloc[-1]['start']
    if equity.iloc[-1].values <= max_equity:
        deltta = equity.index[-1] - date_list[-1]
        start_date = date_list[-1]
        end_date = equity.index[-1]
    else:
        end_date = date_dur.iloc[-1]['end']
    date = start_date.strftime('%Y-%m-%d') + ' - ' + \
        end_date.strftime('%Y-%m-%d')
    logger.debug('---date in analysis---: {}'.format(date))
#     return date
    return date_dur




# ===================================================================================================================
# 11_26_资产走势与策略曲线收益关系分析

# 分析一：思路：多空对冲策略回测部分时段走平原因：是否与价格波动率小有关
location = "C:/Users/workspace/brad_public_workspace_on_win/SH_tongliang/data/11_24_bitmex分钟线_multiplied/bitmex分钟线1124/"

ada_df = pd.read_csv(location+"11_24_4hr_ada.csv", engine="python")
trx_df = pd.read_csv(location+"11_24_4hr_trx.csv", engine="python")
bch_df = pd.read_csv(location+"11_24_4hr_bch.csv", engine="python")
eos_df = pd.read_csv(location+"11_24_4hr_eos.csv", engine="python")
eth_df = pd.read_csv(location+"11_24_4hr_eth.csv", engine="python")
ltc_df = pd.read_csv(location+"11_24_4hr_ltc.csv", engine="python")
xrp_df = pd.read_csv(location+"11_24_4hr_xrp.csv", engine="python")
btc_df = pd.read_csv(location+"11_24_4hr_btc.csv", engine="python")

def delete_unnamed(df):
    del df['Unnamed: 0']
    return df

ada_df = delete_unnamed(ada_df)
trx_df = delete_unnamed(trx_df)
bch_df = delete_unnamed(bch_df)
eos_df = delete_unnamed(eos_df)
eth_df = delete_unnamed(eth_df)
ltc_df = delete_unnamed(ltc_df)
xrp_df = delete_unnamed(xrp_df)
btc_df = delete_unnamed(btc_df)

df = ada_df
time_col = "time"
start_ts = "2018-09-28 00:00:00"
end_ts = "2018-11-02 00:00:00"


def all_period_stddev(df, time_col, backtest_start):
    close = df['close'][df[time_col]>=backtest_start].pct_change()
    stddev = close.std()
    return stddev


def specific_period_stddev(df, time_col, start_ts, end_ts):
    close = df['close'][(df[time_col]>start_ts) & (df[time_col]<end_ts) ].pct_change()
    stddev = close.std()
    return stddev
    
    
def relative_volatility_all_in_one(asst, df, time_col, start_ts, end_ts, backtest_start):
    """
    This "relative_volatility" measures how volatile a specific period of price series is compared to that of all period. 
    """
    specifc_vol = specific_period_stddev(df, time_col, start_ts, end_ts)
    all_vol = all_period_stddev(df, time_col, backtest_start)
    relative_vol = specifc_vol/all_vol
    print ("==============================")
    print (asst)
    print (str((relative_vol)*100)[:5]+"%")
    
    
def relative_volatility_val(asst, df, time_col, start_ts, end_ts, backtest_start):
    """
    This "relative_volatility" measures how volatile a specific period of price series is compared to that of all period. 
    """
    specifc_vol = specific_period_stddev(df, time_col, start_ts, end_ts)
    all_vol = all_period_stddev(df, time_col, backtest_start)
    relative_vol = specifc_vol/all_vol
    return relative_vol


# df = ada_df
time_col = "time"
start_ts = "2018-09-28 00:00:00"
end_ts = "2018-11-02 00:00:00"
backtest_start = "2018-06-29 00:00:00"

relative_volatility_all_in_one("ada", ada_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("bch", bch_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("trx", trx_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("ltc", ltc_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("eos", eos_df, time_col, start_ts, end_ts, backtest_start)
relative_volatility_all_in_one("eth", eth_df, time_col, start_ts, end_ts, backtest_start)

ada_rel_vol = relative_volatility_val("ada", ada_df, time_col, start_ts, end_ts, backtest_start)
bch_rel_vol = relative_volatility_val("bch", bch_df, time_col, start_ts, end_ts, backtest_start)
trx_rel_vol = relative_volatility_val("trx", trx_df, time_col, start_ts, end_ts, backtest_start)
ltc_rel_vol = relative_volatility_val("ltc", ltc_df, time_col, start_ts, end_ts, backtest_start)
eos_rel_vol = relative_volatility_val("eos", eos_df, time_col, start_ts, end_ts, backtest_start)
eth_rel_vol = relative_volatility_val("eth", eth_df, time_col, start_ts, end_ts, backtest_start)
avg_rel_vol= (ada_rel_vol+bch_rel_vol+trx_rel_vol+ltc_rel_vol+eos_rel_vol+eth_rel_vol)/6

print ("==============================")
print ("==============================")
print ("avg_relative_volatility_ratio", str(avg_rel_vol*100)[:5]+"%")

# output:
==============================
ada
74.90%
==============================
bch
46.37%
==============================
trx
85.09%
==============================
ltc
80.47%
==============================
eos
53.62%
==============================
eth
71.93%
==============================
==============================
avg_relative_volatility_ratio 68.73%


# 分析二：回测时期多头不赚钱空头赚钱的原因

def return_analysis(asst, df, time_col, backtest_start):
    close = df['close'][df[time_col]>=backtest_start]
    close_start = close.values[0]
    close_end = close.values[-1]
    avg_daily_return = np.mean(close.pct_change())
    total_return = close_end/close_start - 1
    print (asst)
    print ("avg_daily_return: ", avg_daily_return)
    print ("total_return: ", total_return)
    print ("=============================================")

return_analysis('ada', ada_df, time_col, backtest_start)
return_analysis('eth', eth_df, time_col, backtest_start)
return_analysis('bch', bch_df, time_col, backtest_start)
return_analysis('ltc', ltc_df, time_col, backtest_start)
return_analysis('eos', eos_df, time_col, backtest_start)
return_analysis('trx', trx_df, time_col, backtest_start)

# output:
ada
avg_daily_return:  -0.0007073363288827819
total_return:  -0.5249643366619116
=============================================
eth
avg_daily_return:  -0.000985795217863391
total_return:  -0.6098036485169196
=============================================
bch
avg_daily_return:  -0.0007433045821839179
total_return:  -0.5884917175239756
=============================================
ltc
avg_daily_return:  -0.0005882461890809978
total_return:  -0.4394171779141105
=============================================
eos
avg_daily_return:  -0.0003822458312288239
total_return:  -0.356045162302023
=============================================
trx
avg_daily_return:  -0.0006294343200917069
total_return:  -0.4951923076923077
=============================================

# ===================================================================================================================

# A function that analyzes the performance table from TB regarding which assets gain or lose most:
# Need to put the code together and write into functions
# Notebook to refer to: 11_8_Assets_Performance_Analysis_with_ready-to-use-function

# Import raw performance csv file and build into a data table in a ready-to-process format:
def build_perf_df(location, perf_file):
    perf1 = pd.read_csv(location+perf_file)
    del perf1['Unnamed: 0']
    perf1.columns = ['position_direction', 'asset', 'entry_date', 'entry_price', 
                    'exit_date', 'exit_price', 'qty', 'trade_costs', 'net_gains', 
                    'cum_gains', 'returns', 'cum_returns']
    return perf1  

# Process performance table so that it can be used to generate further asset perf analysis:
def process_perf_df(perf1):
    date_fmt = "%Y-%m-%d"
    perf1['exit_dt'] = perf1['exit_date'].apply(lambda x: datetime.strptime(x, date_fmt))
    perf1['entry_dt'] = perf1['entry_date'].apply(lambda x: datetime.strptime(x, date_fmt))

    perf1['holding_days'] = ' '
    for i, item in enumerate(perf1['holding_days']):
        perf1['holding_days'][i] = perf1['exit_dt'][i] - perf1['entry_dt'][i] 
        perf1['holding_days'][i] =  perf1['holding_days'][i].days

    perf1['returns'] = perf1['returns'].apply(lambda x: float(x.replace("%", "")))
    perf1['returns'] = perf1['returns']/100

    perf1['cum_returns'] = perf1['cum_returns'].apply(lambda x: float(x.replace("%", "")))
    perf1['cum_returns'] = perf1['cum_returns']/100
    
    perf2 = perf1.copy()
    return perf2


def perf_df_analysis(perf2, show_minmax_returns = True, show_return_rank = True):
    perf2['worst_return'] = perf2.groupby(['asset'])['returns'].min()
    perf2['best_return'] = perf2.groupby(['asset'])['returns'].max()
    perf2_rank = perf2.copy()[['asset', 'entry_date', 'exit_date', 'returns']].groupby(['asset']).apply(lambda x: x.sort_values('returns'))
    if show_minmax_returns == True and show_return_rank == True:
                                                                                                        
        print ("worst_return: ", perf2['worst_return'])
        print ("best return: ", perf2['best_return'])
        print ("return ranks", perf2_rank)                                                                                     
    if show_minmax_returns == True and show_return_rank == False:
                                                                                                        
        print ("worst_return: ", perf2['worst_return'])
        print ("best return: ", perf2['best_return'])
    if show_minmax_returns == False and show_return_rank == True:
                                                                                                    
        print ("return ranks: ", perf2_rank)
    else:
        return 







