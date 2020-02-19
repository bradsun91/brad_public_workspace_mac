import pandas as pd, numpy as np
from datetime import datetime
pd.set_option('max_colwidth',200)
import yfinance as yf
import time, urllib
import glob
import os


def sp500():
    sp500_list = ['MMM', 'ABT', 'ABBV', 'ACN', 'ATVI', 'AYI', 'ADBE', 'AMD', 'HII', 'AAP', 'AES', 'AET', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'AGN', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'ANDV', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'AIV', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ARNC', 'AJG', 'AIZ', 'T', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BHGE', 'BLL', 'BAC', 'BK', 'BAX', 'BBT', 'BDX', 'BRK.B', 'BBY', 'BIIB', 'BLK', 'HRB', 'BA', 'BWA', 'BXP', 'BSX', 'BHF', 'BMY', 'AVGO', 'BF.B', 'CHRW', 'CA', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'CBOE', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNC', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHTR', 'CHK', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO', 'GLW', 'COST', 'COTY', 'CCI', 'CSRA', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DOV', 'DWDP', 'DPS', 'DTE', 'DRE', 'DUK', 'DXC', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EVHC', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FISV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FL', 'F', 'FTV', 'FBHS', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GS', 'GT', 'GWW', 'HAL', 'HBI', 'HOG', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HP', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HPQ', 'HUM', 'HBAN', 'IDXX', 'INFO', 'ITW', 'ILMN', 'IR', 'INTC', 'ICE', 'IBM', 'INCY', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IQV', 'IRM', 'JEC', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KHC', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LEG', 'LEN', 'LUK', 'LLY', 'LNC', 'LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'KORS', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'PCAR', 'PKG', 'PH', 'PDCO', 'PAYX', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RJF', 'RTN', 'O', 'RHT', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RCL', 'CRM', 'SBAC', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIG', 'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SPGI', 'SWK', 'SBUX', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYF', 'SNPS', 'SYY', 'TROW', 'TPR', 'TGT', 'TEL', 'FTI', 'TXN', 'TXT', 'TMO', 'TIF', 'TWX', 'TJX', 'TMK', 'TSS', 'TSCO', 'TDG', 'TRV', 'TRIP', 'FOXA', 'FOX', 'TSN', 'UDR', 'ULTA', 'USB', 'UA', 'UAA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'HCN', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS']
    return sp500_list

def all_weather():
    all_weather_portfolio = ["VTI","TLT","IEF","GLD","DBC"]
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
mkt_etf = ["SPY","VXXB","QQQ"]
other_tickers = ["YELP",'UBER','TSLA']
us_sectors = us_sectors_etf()
all_weather = all_weather()
sp500_stocks = sp500()
ticker_list = sp500_stocks+all_weather+us_sectors+mkt_etf+other_tickers

import eventlet
eventlet.monkey_patch()

download_yf_data(start, end, us_db_path, ticker_list)