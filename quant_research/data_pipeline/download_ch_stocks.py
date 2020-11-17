import pandas as pd, numpy as np
from datetime import datetime
pd.set_option('max_colwidth',200)
# import yfinance as yf
import tushare as ts
token='f54f66391ef9cf7675004be86d1ea74740d6df4df971cc10b3bbad91'
ts.set_token(token)
import time, urllib
import glob
import os



def get_data(code,start,end):
    df = ts.pro_bar(ts_code=code, adj='qfq', start_date=start, end_date=end)
    return df

#获取当前交易日最新的股票代码和简称
def get_code():
    codes = pro.stock_basic(list_status='L').ts_code.values
    return codes


def download_tushare_stocks_data(start, end, ch_db_path, ticker_list):
    count = 1
    for ticker in ticker_list:
        if count%200==0:
            print("=======================Sleeping======================")
            time.sleep(60)
        else:
            if not os.path.exists(ch_db_path+ticker+".csv"):
                print("{} is new, start downloading now...".format(ticker))
                try:
                    data = get_data(ticker, start, end)
    #                 print(data)
                    data['trade_date'] = data['trade_date'].astype(str)
                    data['trade_date'] = data['trade_date'].apply(lambda x:x[:4]+"-"+x[4:6]+"-"+x[6:] if len(x)!=10 else x)
                    data.sort_values("trade_date", inplace = True)
                    data.to_csv(ch_db_path+ticker+".csv", index = False)
                    print("{} data file created: {}".format(ticker, end))
                except Exception as e:
                    print(e)

            else:
                print("Already have data csv for {}".format(ticker))
                hist_data = pd.read_csv(ch_db_path+ticker+".csv")   
                hist_data['trade_date'] = hist_data['trade_date'].astype(str)
                hist_data['trade_date'] = hist_data['trade_date'].apply(lambda x:x[:4]+"-"+x[4:6]+"-"+x[6:] if len(x)!=10 else x)
                hist_data.to_csv(ch_db_path+ticker+".csv", index = False)
                hist_data = pd.read_csv(ch_db_path+ticker+".csv")
                try:
                    hist_data_last_date = hist_data['trade_date'].values[-1]        
                    if today > hist_data_last_date:
                        print("Needs to update, start updating new data for {} now...".format(ticker))
                        update_start = hist_data_last_date
                        update_end = today
                        with eventlet.Timeout(60,False):
                            try:
                                new_data = get_data(ticker, update_start, update_end)
                                new_data['trade_date'] = new_data['trade_date'].astype(str)
                                new_data['trade_date'] = new_data['trade_date'].apply(lambda x:x[:4]+"-"+x[4:6]+"-"+x[6:])
                                new_data.to_csv(ch_db_path+ticker+".csv", mode='a', header=False, index = False)
                                updated_duplicated_df = pd.read_csv(ch_db_path+ticker+".csv")
                                updated_df = updated_duplicated_df.drop_duplicates("trade_date")
                                updated_df.sort_values("trade_date", inplace = True)
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


#设置token

pro = ts.pro_api(token)
codes = get_code()
print("Public Company Numbers by today: ", len(codes))

today = str(datetime.now().date())
start = '20100101'
end = today
ch_db_path = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/CH_database/"
ticker_list = codes


import eventlet
import datetime
starttime = datetime.datetime.now()
print ("Executing...")
eventlet.monkey_patch()

download_tushare_stocks_data(start, end, ch_db_path, ticker_list)


print("----------------------------------------------")
endtime = datetime.datetime.now()
duration = (endtime - starttime).seconds
print ("Execution takes {} seconds".format(duration))