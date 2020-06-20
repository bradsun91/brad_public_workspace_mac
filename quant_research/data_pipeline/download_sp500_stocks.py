import pandas as pd, numpy as np
from datetime import datetime
pd.set_option('max_colwidth',200)
import yfinance as yf
import time, urllib
import glob
import os


def all_weather():
    all_weather_portfolio = ["VTI","TLT","IEF","GLD","DBC","SHY","IEI"]
    return all_weather_portfolio

def us_sectors_etf():
    us_sectors = ["IYM","IYZ","FCL","FCD","IYE","IYG","IYH","IYJ","IYR","IYW","IDU"]
    return us_sectors
    
def today_dt():
    today = str(datetime.now().date())
    return today

def download_yf_data(start, end, us_db_path, ticker_list):
    for ticker in ticker_list:
    #     print("testing")
        # Initial downloading:
        if not os.path.exists(us_db_path+ticker+".csv"):
            print("{} is new, start downloading now...".format(ticker))
            with eventlet.Timeout(60,False):
                try:
                    data = yf.download(ticker, start=start, end=end)
                    data.reset_index(inplace = True)
                    data['Ticker'] = ticker
                    data.to_csv(us_db_path+ticker+".csv", index = False)
                    print("{} data file created: {}".format(ticker, end))
                except Exception as e:
                    print(e)

    #         print("Timed Out: Download Failed!")
        # Check for updates:
        else:
            print("Already have data csv for {}".format(ticker))
            hist_data = pd.read_csv(us_db_path+ticker+".csv")
            try:
                hist_data_first_date = hist_data['Date'].values[0]
                if start >= hist_data_first_date:
                    hist_data_last_date = hist_data['Date'].values[-1]
                    if today > hist_data_last_date:
                        print("Needs to update, start updating new data for {} now...".format(ticker))
                        update_start = hist_data_last_date
                        update_end = today
                        with eventlet.Timeout(60,False):
                            try:
                                new_data = yf.download(ticker, start=update_start, end=update_end)
                                new_data.reset_index(inplace = True)
                                new_data['Ticker'] = ticker
                                new_data.to_csv(us_db_path+ticker+".csv", mode='a', header=False, index = False)
                                updated_duplicated_df = pd.read_csv(us_db_path+ticker+".csv")
                                updated_df = updated_duplicated_df.drop_duplicates("Date")
                                updated_df.sort_values("Date", inplace = True)
                                updated_df.to_csv(us_db_path+ticker+".csv", index = False)
                                print("New data updated till today for {}!".format(ticker))
                            except Exception as e:
                                print(e)
            #             print("Timed Out: Update Failed!")
                    else:
                        print("There's no new data to update for {}.".format(ticker))

                else:
                    print("Setup start date earlier than existing data's, trying to pull data from before...")
                    hist_data_last_date = hist_data['Date'].values[-1]
                    if today > hist_data_last_date:
                        print("Needs to update, start updating new data for {} now...".format(ticker))
                        update_start = start
                        update_end = today
                        with eventlet.Timeout(60,False):
                            try:
                                new_data = yf.download(ticker, start=update_start, end=update_end)
                                new_data.reset_index(inplace = True)
                                new_date = new_data['Date'].values[0]
                                print("New data's first pulled date is {}".format(new_date))
                                new_data['Ticker'] = ticker

                                new_data.to_csv(us_db_path+ticker+".csv", mode='a', header=False, index = False)
                                updated_duplicated_df = pd.read_csv(us_db_path+ticker+".csv")
                                updated_df = updated_duplicated_df.drop_duplicates("Date")
                                updated_df.sort_values("Date", inplace = True)
                                updated_df.to_csv(us_db_path+ticker+".csv", index = False)
                                print("New data updated till today for {}!".format(ticker))
                            except Exception as e:
                                print(e)
                    else:
                        print("There's no new data to update for {}.".format(ticker))
            except Exception as e:
                print(e)

    #     print("Data Download/Update for {} is Finished.".format(ticker))
        print("===============================================")
    print("【Updated Finished for today!】")


start = "2005-01-01"
today = today_dt()
end = today
us_db_path = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/US_database/"
mkt_etf = ["SPY","VXXB","QQQ","VXX","^VIX"]
other_tickers = ["YELP",'UBER','TSLA']
us_sectors = us_sectors_etf()
all_weather = all_weather()
data_path = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/quant_research/data_pipeline/"
tickers_df = pd.read_csv(data_path+"most_recent_sp500_tickers.csv")
sp500_tickers = list(tickers_df['sp500_tickers'])
ticker_list = sp500_tickers+all_weather+us_sectors+mkt_etf+other_tickers
# ticker_list = ['ROKU']

import eventlet
eventlet.monkey_patch()

download_yf_data(start, end, us_db_path, ticker_list)