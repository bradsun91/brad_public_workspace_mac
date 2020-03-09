import pandas as pd, numpy as np
np.set_printoptions(suppress=True)# 关掉科学计数法
import glob
import os
import csv
# 一次性merge多个pct_chg
from functools import reduce
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
# import yfinance as yf
import tushare as ts
import time, urllib

from scipy.optimize import minimize



# Get ETF tickers from the summary data sheet
def get_all_etf_tickers_from_summary_file(etf_summary_file):
    ch_etfs_df = pd.read_csv(etf_summary_file, engine="python")
    ch_etfs_df['code'] =ch_etfs_df['证券代码'].apply(lambda x: str(x)[:6])
    ch_etfs_ticker = list(ch_etfs_df['code'].unique())
    return ch_etfs_ticker

#=============================================================
# Get all market data columns as a dataframe from a folder
def get_mkt_data_df(path, ticker_list, date_col):
    # e.g. ch_db_path = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/CH_database/"
    csv_path = path+"*.csv"
    files = glob.glob(csv_path)
    ticker_df_list = []
    for ticker in ticker_list:
        try:
            ticker_df = pd.read_csv(path+ticker+".csv")
            ticker_df = ticker_df.sort_values(date_col)
            ticker_df_list.append(ticker_df) 
        except Exception as e:
            print(e)
    try:
        tickers_data_concated = pd.concat(ticker_df_list)
        tickers_data_concated.reset_index(inplace=True)
        del tickers_data_concated['index']  
    except Exception as e:
        print(e)
    return tickers_data_concated

# Get date_col, price_col, code
def get_date_price_code_df(path, ticker_list, date_col, price_col, code_col):
    # for etf data cols are 'date', 'close', 'code'
    ticker_df_list = []
    for ticker in ticker_list:
        try:
            ticker_df = pd.read_csv(path+ticker+".csv")
            ticker_df = ticker_df.sort_values(date_col)
            ticker_df = ticker_df[[date_col, price_col, code_col]]
            ticker_df_list.append(ticker_df)
        except Exception as e:
            print(e)
    try:
        tickers_data_concated = pd.concat(ticker_df_list)
        tickers_data_concated.reset_index(inplace=True)
        del tickers_data_concated['index']  
    except Exception as e:
        print(e)
    return tickers_data_concated

# Get date_col, price_col, code_col, pct_chg_col
def get_date_price_code_return_df(path, ticker_list, date_col, price_col, code_col):
    # for etf data cols are 'date', 'close', 'code'
    ticker_df_list = []
    for ticker in ticker_list:
        try:
            ticker_df = pd.read_csv(path+ticker+".csv")
            ticker_df = ticker_df.sort_values(date_col)
            ticker_df = ticker_df[[date_col, price_col, code_col]]
            ticker_df['pct_chg'] = ticker_df[price_col].pct_change()
            ticker_df = ticker_df[[date_col, 'pct_chg']].dropna()
            ticker_df['code'] = ticker
            ticker_df_list.append(ticker_df)
        except Exception as e:
            print(e)
    try:
        tickers_data_concated = pd.concat(ticker_df_list)
        tickers_data_concated.reset_index(inplace=True)
        del tickers_data_concated['index']  
    except Exception as e:
        print(e)
    return tickers_data_concated

# Get date_col, price_col, code_col, pct_chg_col
def get_date_price_code_cumreturns_df(path, ticker_list, date_col, price_col, code_col):
    # for etf data cols are 'date', 'close', 'code'
    ticker_df_list = []
    for ticker in ticker_list:
        try:
            ticker_df = pd.read_csv(path+ticker+".csv")
            ticker_df = ticker_df.sort_values(date_col)
            ticker_df = ticker_df[[date_col, price_col, code_col]]
            ticker_df['pct_chg'] = ticker_df[price_col].pct_change()
            ticker_df['cum_returns'] = ticker_df['pct_chg'].cumsum()+1
            ticker_df = ticker_df[[date_col, 'cum_returns']].dropna()
            ticker_df['code'] = ticker
            ticker_df_list.append(ticker_df)
        except:
            pass
    try:
        tickers_data_concated = pd.concat(ticker_df_list)
    except Exception as e:
        print(e)
    return tickers_data_concated

#=============================================================

# Get all market data columns as a dataframe from a folder
def get_mkt_data_list(path, ticker_list, date_col):
    # e.g. ch_db_path = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/CH_database/"
    csv_path = path+"*.csv"
    files = glob.glob(csv_path)
    ticker_df_list = []
    for ticker in ticker_list:
        try:
            ticker_df = pd.read_csv(path+ticker+".csv")
            ticker_df = ticker_df.sort_values(date_col)
            ticker_df_list.append(ticker_df) 
        except Exception as e:
            print(e)
    return ticker_df_list

# Get date_col, price_col, code
def get_date_price_code_list(path, ticker_list, date_col, price_col, code_col):
    # for etf data cols are 'date', 'close', 'code'
    ticker_df_list = []
    for ticker in ticker_list:
        try:
            ticker_df = pd.read_csv(path+ticker+".csv")
            ticker_df = ticker_df.sort_values(date_col)
            ticker_df = ticker_df[[date_col, price_col]]
            ticker_df.columns = [date_col, ticker]
            ticker_df_list.append(ticker_df)
        except Exception as e:
            print(e)
    return ticker_df_list

# Get date_col, price_col, code_col, pct_chg_col
def get_date_price_code_return_list(path, ticker_list, date_col, price_col, code_col):
    # for etf data cols are 'date', 'close', 'code'
    ticker_df_list = []
    for ticker in ticker_list:
        try:
            ticker_df = pd.read_csv(path+ticker+".csv")
            ticker_df = ticker_df.sort_values(date_col)
            ticker_df = ticker_df[[date_col, price_col, code_col]]
            ticker_df['pct_chg'] = ticker_df[price_col].pct_change()
            ticker_df = ticker_df[[date_col, 'pct_chg']].dropna()
            ticker_df.columns = [date_col, ticker]
            ticker_df_list.append(ticker_df)
        except Exception as e:
            print(e)
    return ticker_df_list

# Get date_col, price_col, code_col, pct_chg_col
def get_date_price_code_cumreturns_list(path, ticker_list, date_col, price_col, code_col):
    # for etf data cols are 'date', 'close', 'code'
    ticker_df_list = []
    for ticker in ticker_list:
        try:
            ticker_df = pd.read_csv(path+ticker+".csv")
            ticker_df = ticker_df.sort_values(date_col)
            ticker_df = ticker_df[[date_col, price_col, code_col]]
            ticker_df['pct_chg'] = ticker_df[price_col].pct_change()
            ticker_df['cum_returns'] = ticker_df['pct_chg'].cumsum()+1
            ticker_df = ticker_df[[date_col, 'cum_returns']].dropna()
            ticker_df.columns = [date_col, ticker]
            ticker_df_list.append(ticker_df)
        except:
            pass
    return ticker_df_list

#=============================================================

def get_sector_leaders():
    etf_sectors = pd.read_csv("/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/ETF_板块已分类.csv")
#     etf_sectors.dropna(inplace=True)
    filtered_1 = etf_sectors[etf_sectors['所属板块']!=""]
    filtered_1.columns = ['ticker','name','type1','type2','mkt_cap','institution_holdings',
                          'ins_holding_pct','fee1%','fee2%','fee3%','sector']
    filtered_1['mkt_cap'] = filtered_1['mkt_cap'].apply(lambda x: x.replace(",","")).apply(lambda x: float(x))
    filtered_1.reset_index(inplace = True)
    del filtered_1['index']
    filtered_1['rank'] = filtered_1.groupby(['sector'])['mkt_cap'].rank(ascending = False)
    filtered_2 = filtered_1[filtered_1['rank']==1]
    filtered_2['code'] = filtered_2['ticker'].str.split(".", expand = True)[0]
    filtered_2['exchange'] = filtered_2['ticker'].str.split(".", expand = True)[1]
    filtered_2_tickers = list(filtered_2['code'].unique())
    return filtered_2_tickers




def merge_df_for_reduce(df1, df2, date_col="trade_date"):
    # By default the etf's date_col goes by 'date'
    merged = df1.merge(df2, on = date_col, how = 'outer')
    merged.sort_values(date_col, inplace = True)
    return merged

# merge a list of dfs on date_col
def merge_dfs_by_ticker(ticker_df_list, date_col):
    merged_all = reduce(merge_df_for_reduce, ticker_df_list)
    merged_all.set_index(date_col, inplace=True)
    merged_all.dropna(how="all", axis = 1, inplace = True)
    merged_all.fillna(method="ffill", inplace = True)
    return merged_all


def MACD(df, n_fast, n_slow, n_macd, price_col): # n_fast = 12, n_slow = 26
    """
    http://stockcharts.com/docs/doku.php?id=scans:indicators
    MACD, MACD Signal and MACD difference, rationale CHECKED, code CHECKED, updated
    # Conventional look-back window for calculating MACDsign is 9
    """
    EMAfast = df[price_col].ewm(span = n_fast, min_periods = n_fast - 1).mean()
    EMAslow = df[price_col].ewm(span = n_slow, min_periods = n_slow - 1).mean()
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = MACD.ewm(span = n_macd, min_periods = n_macd-1).mean().rename('MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df['MACD_Diff'] = MACD
    df['MACD_Diff_EMA'] = MACDsign
    df['MACD'] = MACDdiff
    df['SIGNAL_STATUS'] = df['MACD'].apply(lambda x: "多头状态" if x>0 else ("空头状态" if x<0 else "无信号状态"))
    return df

def calc_macd_signals(tickers_data_concated, ticker_list, code_col, ticker_type, price_col):
    signal_record = []
    signal_data = []
    if len(ticker_list)!=1:
        for ticker in ticker_list:
            try:
                if ticker_type == "float":
                # Be aware of types of ticker values here, whether it's float or strings, depends.
                    single_ticker_df = tickers_data_concated[tickers_data_concated[code_col]==float(ticker)]
                elif ticker_type == "string":
                    single_ticker_df = tickers_data_concated[tickers_data_concated[code_col]==ticker]
                signal_df = MACD(single_ticker_df, 12, 26, 9, price_col)
#                 last_signal = signal_df[[code_col,'SIGNAL_STATUS']].values[-1]
                signal_data.append(signal_df)
#                 signal_record.append(last_signal)
            except:
                pass
        signal_data_df = pd.concat(signal_data)
    else:
        try:                
            signal_df = MACD(tickers_data_concated, 12, 26, 9, price_col)
        except:
            pass
        signal_data_df = signal_df
    return signal_data_df

def make_numeric_signals(series):
    for item in series:
        if item =="多":
            return 1
        elif item =="空":
            return -1
        else:
            return 0
        
def get_last_signals_macd(signal_data_df, date_col, code_col):
    # v1 is the version of generating the og macd signals
    signal_data_df['SIGNAL_DIRECTION'] = signal_data_df['SIGNAL_STATUS'].apply(lambda x: make_numeric_signals(x))
    signal_data_df['SIGNAL_DIRECTION_DIFF'] = signal_data_df.groupby([code_col])['SIGNAL_DIRECTION'].diff()
    signal_data_df['SIGNAL_ACTION'] = signal_data_df['SIGNAL_DIRECTION_DIFF'].apply(lambda x: "LONG" if x==2 else("SHORT" if x==-2 else "NO CHANGE"))
    most_recent_signals = signal_data_df.groupby([code_col])[[date_col,code_col,'SIGNAL_STATUS','SIGNAL_ACTION']].tail(1)
    return most_recent_signals

def merge_current_pos_with_target_pos(etf_path, cur_positions, tgt_most_recent_etf_macd_signals, code_col, date_col, code_type, price_col):
    cur_pos_prices = get_mkt_data_df(etf_path, cur_positions, date_col)
    cur_pos_etf_macd_signals = calc_macd_signals(cur_pos_prices, cur_positions, code_col, code_type,price_col)
    most_recent_cur_pos_etf_macd_signals = get_last_signals_macd(cur_pos_etf_macd_signals, date_col, code_type)
    most_recent_cur_pos_etf_macd_signals['TYPE'] = 'CUR_POS'
    all_macd_signal_df = most_recent_cur_pos_etf_macd_signals.merge(tgt_most_recent_etf_macd_signals, on = [date_col,code_col], how = 'outer')
    return all_macd_signal_df

def get_smart_weight(pct, method='risk parity', cov_adjusted=False, wts_adjusted=False):
    if cov_adjusted == False:
        #协方差矩阵
        cov_mat = pct.cov()
    else:
        #调整后的半衰协方差矩阵
        cov_mat = pct.iloc[:len(pct)/4].cov()*(1/10.) + pct.iloc[len(pct)/4+1:len(pct)/2].cov()*(2/10.) +\
            pct.iloc[len(pct)/2+1:len(pct)/4*3].cov()*(3/10.) + pct.iloc[len(pct)/4*3+1:].cov()*(4/10.)
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError('cov_mat should be pandas DataFrame！')
        
    omega = np.matrix(cov_mat.values)  # 协方差矩阵

    a, b = np.linalg.eig(np.array(cov_mat)) #a为特征值,b为特征向量
    a = np.matrix(a)
    b = np.matrix(b)
    # 定义目标函数
    
    def fun1(x):
        tmp = (omega * np.matrix(x).T).A1
        risk = x * tmp/ np.sqrt(np.matrix(x) * omega * np.matrix(x).T).A1[0]
        delta_risk = [sum((i - risk)**2) for i in risk]
        return sum(delta_risk)
    
    def fun2(x):
        tmp = (b**(-1) * omega * np.matrix(x).T).A1
        risk = (b**(-1)*np.matrix(x).T).A1 * tmp/ np.sqrt(np.matrix(x) * omega * np.matrix(x).T).A1[0]
        delta_risk = [sum((i - risk)**2) for i in risk]
        return sum(delta_risk)
    
    # 初始值 + 约束条件 
    x0 = np.ones(omega.shape[0]) / omega.shape[0]  
    bnds = tuple((0,None) for x in x0)
    cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
    options={'disp':False, 'maxiter':1000, 'ftol':1e-20}
        
    if method == 'risk parity':
        res = minimize(fun1, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == 'pc risk parity':
        res = minimize(fun2, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    else:
        raise ValueError('method error！！！')
        
    # 权重调整
    if res['success'] == False:
        # print res['message']
        pass
    wts = pd.Series(index=cov_mat.index, data=res['x'])
    
    if wts_adjusted == True:
        wts[wts < 0.0001]=0.0
        wts = wts / wts.sum()
    elif wts_adjusted == False:
        wts = wts / wts.sum()
    else:
        raise ValueError('wts_adjusted should be True/False！')
        
    risk = pd.Series(wts * (omega * np.matrix(wts).T).A1 / np.sqrt(np.matrix(wts) * omega * np.matrix(wts).T).A1[0],index = cov_mat.index)
    risk[risk<0.0] = 0.0
    return wts,risk

def get_df_wts(tgt_merged_returns):
    wts, risk = get_smart_weight(tgt_merged_returns, method='risk parity', cov_adjusted=False, wts_adjusted=False)
    df_wts = pd.DataFrame(wts)
    df_wts.reset_index(inplace = True)
    df_wts.columns = ['ETF', 'Portfolio_Weight']
    etfs = list(df_wts['ETF'])
    weights = list(df_wts['Portfolio_Weight'])
    return df_wts, etfs, weights

def draw_risk_parity_pie(weights, etfs):
    # 保证圆形
    plt.figure(1, figsize = (25, 25))
    plt.axes(aspect=1)
    plt.pie(x=weights, labels=etfs, autopct='%3.1f %%')
    plt.title("Risk-Parity Allocation", fontsize = 15)
    plt.show()


# tgt_returns = get_date_price_code_return_list(etf_path, etf_pool, 'date', 'close', 'code')
# tgt_cum_returns = get_date_price_code_cumreturns_list(etf_path, etf_pool, 'date', 'close', 'code')
# tgt_merged_returns = merge_dfs_by_ticker(tgt_returns, "date")
# tgt_merged_cumreturns = merge_dfs_by_ticker(tgt_cum_returns, "date")

# df_wts, etfs, weights = get_df_wts(tgt_merged_returns)
# draw_risk_parity_pie(weights, etfs)

# Filter lowest correlations
def select_N_lowest_corr_assets(merged_all, asset_category, n):
    # asset_category be default for etf is 'etf'
    # Create ETFs' correlation matrix dataframe
    merged_all_corr = merged_all.corr()
    merged_all_corr_abs = abs(merged_all_corr)
    corr_mean_dict = {}
    for row in merged_all_corr_abs.iterrows():
        row_list = list(row)
        print(row_list[0])
        print(np.mean(row_list[1]))
        corr_mean_dict[row_list[0]] = np.mean(row_list[1])
        print("========")
    corr_mean_df = pd.DataFrame()
    corr_mean_df[asset_category] = corr_mean_dict.keys()
    corr_mean_df['abs_corr_mean'] = corr_mean_dict.values()
    corr_mean_df_assets = corr_mean_df.sort_values("abs_corr_mean", ascending=True)
    lowest_corr_assets_list = list(corr_mean_df_assets[asset_category])[:n]
    return lowest_corr_assets_list, corr_mean_df_assets

#### From SBTV1
####  =======================================================================================================================================




