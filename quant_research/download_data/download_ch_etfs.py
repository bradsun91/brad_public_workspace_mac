import pandas as pd, numpy as np
from datetime import datetime
# import yfinance as yf
import tushare as ts
import time, urllib
ts.set_token('2f31c3932ead9fcc3830879132cc3ec8df3566550f711889d4a30f67')
import glob
import os


def handle_data_files_return_tickers(file_path):
    ch_etfs_df = pd.read_csv(file_path)
    ch_etfs_df.dropna(inplace = True)
    ch_etfs_df['基金规模\n[单位] 元'] = ch_etfs_df['基金规模\n[单位] 元'].apply(lambda x: float(x.replace(",","")))
    ch_etfs_df['机构投资者持有份额\n[报告期] 2019中报\n[单位] 份'] = ch_etfs_df['机构投资者持有份额\n[报告期] 2019中报\n[单位] 份'].apply(lambda x: float(x.replace(",","")))
    ch_etfs_df.sort_values("基金规模\n[单位] 元", ascending=False, inplace =True)
    ch_etfs_df = ch_etfs_df
    ch_etfs = ch_etfs_df.copy()
    ch_etfs['code'] =ch_etfs['证券代码'].apply(lambda x: str(x)[:6])
    ch_etfs_ticker = list(ch_etfs['code'].unique())
    return ch_etfs_ticker

def get_etf_data(Code, BeginTime, EndTime):
    df = ts.get_k_data(Code, index = False,  start = BeginTime, end = EndTime)
    return df


def download_tushare_etf_data(start, end, ch_db_path, ticker_list):
    count = 1
    for ticker in ticker_list:
        if count%200==0:
            print("=======================Sleeping======================")
            time.sleep(60)
        else:
            if not os.path.exists(ch_db_path+ticker+".csv"):
                print("{} is new, start downloading now...".format(ticker))
                try:
                    ############ replace get_data with other customized functions #############
                    data = get_etf_data(ticker, start, today)
                    ############ replace get_data with other customized functions #############
    #                 print(data)
                    data['date'] = data['date'].astype(str)
    #                 data['date'] = data['date'].apply(lambda x:x[:4]+"-"+x[4:6]+"-"+x[6:] if len(x)!=10 else x)
                    data.sort_values("date", inplace = True)
                    data.to_csv(ch_db_path+ticker+".csv", index = False)
                    print("{} data file created: {}".format(ticker, end))
                except Exception as e:
                    print(e)

            else:
                print("Already have data csv for {}".format(ticker))
                hist_data = pd.read_csv(ch_db_path+ticker+".csv")   
                hist_data['date'] = hist_data['date'].astype(str)
                hist_data.to_csv(ch_db_path+ticker+".csv", index = False)
                hist_data = pd.read_csv(ch_db_path+ticker+".csv")
                try:
                    hist_data_last_date = hist_data['date'].values[-1]        
                    if today > hist_data_last_date:
                        print("Needs to update, start updating new data for {} now...".format(ticker))
                        update_start = hist_data_last_date
                        update_end = today
                        with eventlet.Timeout(60,False):
                            try:
                                ############ replace get_data with other customized functions #############
                                new_data = get_etf_data(ticker, update_start, update_end)
                                ############ replace get_data with other customized functions #############

                                new_data['date'] = new_data['date'].astype(str)
                                new_data.to_csv(ch_db_path+ticker+".csv", mode='a', header=False, index = False)
                                updated_duplicated_df = pd.read_csv(ch_db_path+ticker+".csv")
                                updated_df = updated_duplicated_df.drop_duplicates("date")
                                updated_df.sort_values("date", inplace = True)
                                updated_df.to_csv(ch_db_path+ticker+".csv", index = False)
                                print("New data updated till today for {}!".format(ticker))
                            except Exception as e:
                                print(e)
            #             print("Timed Out: Update Failed!")
                    else:
                        print("There's no new data to update for {}.".format(ticker))
                except Exception as e:
                    print(e)

        #     print("Data Download/Update for {} is Finished.".format(ticker))
            print("=======================Executed: {}=======================".format(count))
        count+=1
    print("【Updated Finished for today!】")

today = str(datetime.now().date())
start = '2010-01-01'
end = today
etf_ticker_path = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/CH_ETFs.csv"
etf_path_to_csv = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/CH_database/"
ticker_list = handle_data_files_return_tickers(etf_ticker_path)

import eventlet
eventlet.monkey_patch()

download_tushare_etf_data(start, end, etf_path_to_csv, ticker_list)