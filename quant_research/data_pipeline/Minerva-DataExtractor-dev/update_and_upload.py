import datetime as dt 
import pandas as pd
import yfinance as yf
yf.pdr_override()
import fire 
from pandas_datareader import data as pdr
from fredapi import Fred
import quandl
import json
import numpy as np

from sp500.aws_boto3 import upload_to_aws
from sp500.Databases_Design_JSON import sp500_financial
from sp500.fetch_options import get_options
from sp500.save_tickers import save_sp500_tickers
from sp500_news_crawler.news import sp500_news
from cryptocurrency.fetch_crypto import Crypto
from futures.fetch_futures import Futures
from indexes.fetch_index import Index

class Upload(object):
    def daily(self):
        print('start!')
        # get key statistics and upload to s3
        key = sp500_financial('key_statistics')
        key.get_key_stats_df()

        upload_to_aws('sp500_key_statistics_dfs/{}_key_statistics.json'.format(str(dt.date.today()-dt.timedelta(days=1))), 'moyi-minerva', 'sp500_key_statistics_dfs/{}_key_statistics.json'.format(str(dt.date.today()-dt.timedelta(days=1))))
                
        # get market cap and upload
        market_cap = sp500_financial('market_cap')
        market_cap.get_market_cap_df()
        upload_to_aws('sp500_market_cap_dfs/{}_market_cap.json'.format(str(dt.date.today()-dt.timedelta(days=1))), 'moyi-minerva', 'sp500_market_cap_dfs/{}_market_cap.json'.format(str(dt.date.today()-dt.timedelta(days=1))))

        # get options and upload 
        get_options(option_type='put')
        get_options(option_type='call')
        upload_to_aws('sp500_put/{}_put.json'.format(str(dt.date.today()-dt.timedelta(days=1))), 'moyi-minerva', 'sp500_put/{}_put.json'.format(str(dt.date.today()-dt.timedelta(days=1))))
        upload_to_aws('sp500_call/{}_call.json'.format(str(dt.date.today()-dt.timedelta(days=1))), 'moyi-minerva', 'sp500_call/{}_call.json'.format(str(dt.date.today()-dt.timedelta(days=1))))

        # get daily market data and upload
        today=dt.date.today()
        d=today-dt.timedelta(days=1)
        yesterday=d.strftime("%Y_%m_%d")

        company_list=save_sp500_tickers()

        # add indexes ETF to market data
        company_list.extend(['SPY','VXX','UVXY'])

        if (d.weekday() >=0) and (d.weekday() <= 4):
            for ticker in company_list:
                ticker = ticker.replace('.','')
                data=yf.download(ticker, period="1 d", auto_adjust=False)
                data['ticker'] = ticker

                def date_process(date):
                    return str(date).split()[0]

                def float_process(num):
                    return float(num)

                data.reset_index(inplace=True)
                data.rename(columns = {'Date': 'date', 'Open':'open', 'Close':'close', 'High':'high', 'Low':'low', 'Adj Close':'adj_close', 'Volume':'volume'}, inplace=True)
                
                data['date'] = data['date'].apply(date_process)
                data['open'] = data['open'].apply(float_process)
                data['close'] = data['close'].apply(float_process)
                data['high'] = data['high'].apply(float_process)
                data['low'] = data['low'].apply(float_process)
                data['adj_close'] = data['adj_close'].apply(float_process)
                data = data.replace({np.nan:None})
                data = data.to_dict('records')
                data = [json.dumps(df) for df in data]
                data = '\n'.join(data)

                # Write to JSON
                with open('{}_market_data.json'.format(yesterday),'a') as outfile:
                    outfile.write(data + '\n')

            upload_to_aws('{}_market_data.json'.format(yesterday), 'moyi-minerva', 'sp500_market_data/{}_market_data.json'.format(yesterday))

        #daily economic indicators
        indicators_code=["GDP","CPIAUCSL","UNRATE","GFDEBTN","GFDEGDQ188S","INDPRO","TCU","DTWEXM","USSLIND","MEHOINUSA672N","PCE","PSAVERT","T5YIE"]
        indicators={"GDP":"Gross Domestic Product","CPIAUCSL":"Consumer Price Index for All Urban Consumers: All Items","UNRATE":"Civilian Unemployment Rate","GFDEBTN":"Total Public Debt","GFDEGDQ188S":"Total Public Debt as Percent of Gross Domestic Product","INDPRO":"Industrial Production Index","TCU":"Capacity Utilization: Total Industry","DTWEXM":"Trade Weighted U.S. Dollar Index: Major Currencies","USSLIND":"Leading Index for the United States","MEHOINUSA672N":"Real Median Household Income in the United States","PCE":"Personal Consumption Expenditures","PSAVERT":"Personal Saving Rate","T5YIE":"5-Year Breakeven Inflation Rate"}

        for i in indicators_code:
            data=quandl.get("FRED/{}".format(i), authtoken="cHu9vuS9yVzx2Y6oy7Zm")
            data.reset_index(level=0, inplace=True)
            last=float(data['Value'].iloc[-1])
            reference=data['Date'].iloc[-1].strftime("%B/%d/%Y")
            minimum=float(min(data['Value']))
            maximum=float(max(data['Value']))
            td =(data['Date'].iloc[-1]-data['Date'].iloc[-2]).days
            if td > 85 and td <95:
                Frequency="Quarterly"
            elif td>0 and td <5:
                Frequency="Daily"
            elif td > 25 and td <35:
                Frequency="Monthly"
            elif td >360 and td < 370 :
                Frequency="Yearly"
            indicator=indicators[i]
            date=yesterday
            df = {'indicator':indicator,'last':last,'reference':reference,'min':minimum,'max':maximum,'frequency':Frequency,'date':date}
            df = json.dumps(df)

            with open('{}_economic_indicators.json'.format(yesterday),'a') as outfile:
                outfile.write(df + '\n')

        upload_to_aws("{}_economic_indicators.json".format(yesterday), 'moyi-minerva', 'economic_indicators/{}_economic_indicators.json'.format(yesterday))

    def quarter(self):
        balance_quarter = sp500_financial('balance',frequency='quarterly')
        cash_quarter = sp500_financial('cash',frequency='quarterly')
        income_quarter = sp500_financial('income',frequency='quarterly')

        balance_quarter.get_stmts_df()
        cash_quarter.get_stmts_df()
        income_quarter.get_stmts_df()

        for name in ('balance','cash','income'):
            if name == 'cash':
                upload_to_aws('sp500_fundamentals_dfs/{}_{}_quarterly.json'.format(str(dt.date.today()), name), 'moyi-minerva', 'sp500_{}_quarterly/{}_{}_quarterly.json'.format('cashflow', str(dt.date.today()), name)) 
            else:
                upload_to_aws('sp500_fundamentals_dfs/{}_{}_quarterly.json'.format(str(dt.date.today()), name), 'moyi-minerva', 'sp500_{}_quarterly/{}_{}_quarterly.json'.format(name, str(dt.date.today()), name))

    def annual(self):
        # get financial statements and upload to S3
        balance_annual = sp500_financial('balance',frequency='annual')
        cash_annual = sp500_financial('cash',frequency='annual')
        income_annual = sp500_financial('income',frequency='annual')
        balance_annual.get_stmts_df()
        cash_annual.get_stmts_df()
        income_annual.get_stmts_df()

        for name in ('balance','cash','income'):
            if name == 'cash':
                upload_to_aws('sp500_fundamentals_dfs/{}_{}_annual.json'.format(str(str(dt.date.today())), name), 'moyi-minerva', 'sp500_{}_annual/{}_{}_annual.json'.format('cashflow', str(dt.date.today()), name))
            else:
                upload_to_aws('sp500_fundamentals_dfs/{}_{}_annual.json'.format(str(dt.date.today()), name), 'moyi-minerva', 'sp500_{}_annual/{}_{}_annual.json'.format(name, str(dt.date.today()), name))
    
    def crypto_daily(self):
        # initiate Crypto object and get daily prices
        data = Crypto()
        data.get_crypto()

        # upload to s3
        upload_to_aws('crypto_data/{}_crypto_data.json'.format(str(dt.date.today()-dt.timedelta(days=1))), 'moyi-minerva', 'crypto_data/{}_crypto_data.json'.format(str(dt.date.today()-dt.timedelta(days=1))))

    def futures_daily(self):
        d = dt.date.today()-dt.timedelta(days=1)
        if (d.weekday() >=0) and (d.weekday() <= 4):
            Futures().get_futures()
        
            # upload to s3
            upload_to_aws('futures_data/{}_futures_data.csv'.format(str(d)),'moyi-minerva','futures_data/{}_futures_data.csv'.format(str(d)))
    
    def index_daily(self):
        # intiate index object and get daily major indexes
        d = dt.date.today()-dt.timedelta(days=1)
        if (d.weekday() >=0) and (d.weekday() <= 4):
            Index().get_indexes()

            #upload to s3
            upload_to_aws('indexes_data/{}_indexes.json'.format(str(d)), 'moyi-minerva', 'indexes_data/{}_indexes.json'.format(str(d)))

    def news_daily(self):   
        robots = sp500_news()
        robots.run()

    def upload_news(self):
        upload_to_aws('sp500_news_crawler/Nasdaq_news_{}.csv'.format(str(dt.date.today().strftime('%Y-%m-%d'))), 'moyi-minerva', 'news/Nasdaq_news_{}.csv'.format(str(dt.date.today().strftime('%Y-%m-%d'))))
