import os
import glob
from datetime import datetime
import pandas as pd, numpy as np
pd.set_option('max_colwidth',200)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

#读取数据
ch_etfs_df = pd.read_csv("/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/"+"CH_ETFs.csv")
ch_etfs_df.dropna(inplace = True)
ch_etfs_df['基金规模\n[单位] 元'] = ch_etfs_df['基金规模\n[单位] 元'].apply(lambda x: float(x.replace(",","")))
ch_etfs_df['机构投资者持有份额\n[报告期] 2019中报\n[单位] 份'] = ch_etfs_df['机构投资者持有份额\n[报告期] 2019中报\n[单位] 份'].apply(lambda x: float(x.replace(",","")))
ch_etfs_df.sort_values("基金规模\n[单位] 元", ascending=False, inplace =True)
ch_etfs_df = ch_etfs_df.head(50)
ch_etfs = ch_etfs_df.copy()
ch_etfs['code'] =ch_etfs['证券代码'].apply(lambda x: str(x)[:6])
ch_etfs_ticker = list(ch_etfs['code'].unique())
# Add the sp500 etf
etf_tickers = ['513500']+ch_etfs_ticker


"""
1. Correlation Analysis for ETFs and Filter Targets
Function 1. Plot correlation matrix heatmap of multiple assets from: to-be-made func: etf_corr_heatmap(ticker, list, path, price_col, date_col)

Function 2. Filter assets with lowest correlations among them
"""

# This part is for generating correlation heatmaps

today = str(datetime.now().date())
start = '2010-01-01'
end = today

ch_db_path = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/CH_database/"
ticker_df_list = []
for ticker in etf_tickers:
    print("Reading ETF: ", ticker)
    try:
        ticker_df = pd.read_csv(ch_db_path+ticker+".csv")
        ticker_df = ticker_df.sort_values('date')
        ticker_df = ticker_df[['date','close','code']]
        ticker_df['pct_chg'] = ticker_df['close'].pct_change()
        ticker_df = ticker_df[['date', 'pct_chg']].dropna()
        ticker_df.columns = ['date',ticker]
        ticker_df['date'] = pd.to_datetime(ticker_df['date'])
    #     stock.set_index('date', inplace=True)
        ticker_df_list.append(ticker_df)
#     except FileNotFoundError:
#         print("Does not have ETF {}, start downloading now...".format(ticker))
#         data = GetHistoryData(ticker, start, today)
#         data.to_csv(ch_db_path+ticker+".csv", index = False)
#         print("ETF {} downloaded.".format(ticker))
    except:
        pass


# 一次性merge多个pct_chg
from functools import reduce

# 先为之后使用reduce铺路：创造一个merge的函数：
def merge_df(df1, df2):
    df1.sort_values('date', inplace = True)
    merged = df1.merge(df2, on = 'date', how = 'outer')
    merged.sort_values('date', inplace = True)
    return merged

# 重要知识点：reduce
# stock_list里都是一个个dataframe
merged_all = reduce(merge_df, ticker_df_list)
merged_all.set_index('date', inplace=True)
merged_all.dropna(how="all", axis = 1, inplace = True)



# plot heatmap:
fig, ax = plt.subplots(figsize = (15, 8))
sns.heatmap(merged_all.corr()[abs(merged_all.corr())>-2], ax = ax, cmap = 'Blues', vmax = 1.0, vmin = -1.0, annot=True)
plt.xlabel('ETF', fontsize = 10)
plt.ylabel('ETF', fontsize = 10)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10);


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


# 计算平均相关性

corr_mean_df = pd.DataFrame()
corr_mean_df['etf'] = corr_mean_dict.keys()
corr_mean_df['abs_corr_mean'] = corr_mean_dict.values()
corr_mean_df_10_etf = corr_mean_df.sort_values("abs_corr_mean", ascending=True)

# 选取和其他资产平均相关性最低的10只

selected_10_etf_list = list(corr_mean_df_10_etf['etf'])[:10]
selected_10_etf_list