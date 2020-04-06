from aws_boto3 import upload_to_aws
from Databases_Design_JSON import sp500_financial
# from fetch_options import get_options
import datetime as dt 
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
from save_tickers import save_sp500_tickers

# '''# get financial statements and upload to S3
# balance_annual = sp500_financial('balance',frequency='annual')
# cash_annual = sp500_financial('cash',frequency='annual')
# income_annual = sp500_financial('income',frequency='annual')
# balance_quarter = sp500_financial('balance',frequency='quarterly')
# cash_quarter = sp500_financial('cash',frequency='quarterly')
# income_quarter = sp500_financial('income',frequency='quarterly')

# balance_annual.get_stmts_df()
# cash_annual.get_stmts_df()
# income_annual.get_stmts_df()
# balance_quarter.get_stmts_df()
# cash_quarter.get_stmts_df()
# income_quarter.get_stmts_df()

# for name in ('balance','cash','income'):
#         if name == 'cash':
#                 uploaded_annual = upload_to_aws('sp500_fundamentals_dfs/{}_{}_annual.json'.format(str(str(dt.date.today()-dt.timedelta(days=1))), name), 'moyi-minerva', 'sp500_{}_annual/{}_{}_annual.json'.format('cashflow', str(dt.date.today()-dt.timedelta(days=1)), name))
#                 uploaded_quarterly = upload_to_aws('sp500_fundamentals_dfs/{}_{}_quarterly.json'.format(str(dt.date.today()-dt.timedelta(days=1)), name), 'moyi-minerva', 'sp500_{}_quarterly/{}_{}_quarterly.json'.format('cashflow', str(dt.date.today()-dt.timedelta(days=1)), name)) 
#         else:
#                 uploaded_annual = upload_to_aws('sp500_fundamentals_dfs/{}_{}_annual.json'.format(str(dt.date.today()-dt.timedelta(days=1)), name), 'moyi-minerva', 'sp500_{}_annual/{}_{}_annual.json'.format(name, str(dt.date.today()-dt.timedelta(days=1)), name))
#                 uploaded_quarterly = upload_to_aws('sp500_fundamentals_dfs/{}_{}_quarterly.json'.format(str(dt.date.today()-dt.timedelta(days=1)), name), 'moyi-minerva', 'sp500_{}_quarterly/{}_{}_quarterly.json'.format(name, str(dt.date.today()-dt.timedelta(days=1)), name))
# print ('\nsuccessfully uploaded all financial statements!!\n')'''

# # get key statistics and upload to s3
# key = sp500_financial('key_statistics')
# key.get_key_stats_df()

# uploaded_key_stats = upload_to_aws('sp500_key_statistics_dfs/{}_key_statistics.json'.format(str(dt.date.today()-dt.timedelta(days=1))), 'moyi-minerva', 'sp500_key_statistics_dfs/{}_key_statistics.json'.format(str(dt.date.today()-dt.timedelta(days=1))))
# print ('\nsuccessfully uploaded all key statistics!!\n')

# # get market cap and upload
# market_cap = sp500_financial('market_cap')
# market_cap_df = market_cap.get_market_cap_df()

# uploaded_market_cap = upload_to_aws('sp500_market_cap_dfs/{}_market_cap.csv'.format(str(dt.date.today()-dt.timedelta(days=1))), 'moyi-minerva', 'sp500_market_cap_dfs/{}_market_cap.csv'.format(str(dt.date.today()-dt.timedelta(days=1))))
# print ('\nsuccessfully uploaded all market capitalizations!!\n')

# # get options and upload 
# puts = get_options(option_type='put')
# calls = get_options(option_type='call')

# upload_puts = upload_to_aws('sp500_put/{}_put.csv'.format(str(dt.date.today()-dt.timedelta(days=1))), 'moyi-minerva', 'sp500_put/{}_put.csv'.format(str(dt.date.today()-dt.timedelta(days=1))))
# upload_calls = upload_to_aws('sp500_call/{}_call.csv'.format(str(dt.date.today()-dt.timedelta(days=1))), 'moyi-minerva', 'sp500_call/{}_call.csv'.format(str(dt.date.today()-dt.timedelta(days=1))))

# '''uploaded_ticker_secotr = upload_to_aws('sp500_tickers_sectors.csv', 'moyi-minerva', 'sp500_tickers_sectors.csv')
# print ('\nsuccessfully uploaded all company industry!!\n')'''

# #get daily market data and upload
# today=dt.date.today()
# d=today-dt.timedelta(days=1)
# yesterday=d.strftime("%Y_%m_%d")

# company_list=save_sp500_tickers()
# df=pd.DataFrame()
# if (d.weekday() >=1) and (d.weekday() <= 5):
#     for i in company_list:
#         data=yf.download(i, period="1 d",auto_adjust=False)
#         data['Ticker']=i
#         data.columns =['open','high','low','close','adj_close','volume','ticker']
#         df=pd.concat([df,data])

# df.to_csv("market_data_{}.csv".format(yesterday))
# uploaded_market_data = upload_to_aws('market_data_{}.csv'.format(yesterday), 'moyi-minerva', 'sp500_market_data/market_data_{}.csv'.format(yesterday))
upload_to_aws('sp500_news_crawler/Nasdaq_news.csv', 'moyi-minerva', 'sp500_news/{}_news_Nasdaq.csv'.format(str(dt.date.today()-dt.timedelta(days=1))))