from yahoo_fin import options
import pandas as pd
import datetime as dt
import os
import json
import numpy as np

from .save_tickers import save_sp500_tickers

def get_options(tickers=None, option_type=None, date=None):
    if not tickers:
        tickers = save_sp500_tickers()

    # add indexes options to tickers list
    tickers.extend(['SPY', 'VXX', 'UVXY'])
    
    print('# start fetching {} options'.format(option_type))
    for i, ticker in enumerate(tickers):
        ticker = ticker.replace('.','')
        # get all put option data
        if option_type == 'put':
            try:
                df = options.get_puts(ticker)
            except:
                print('Failed:', ticker)
                continue
        # get all call option data 
        elif option_type == 'call':
            try:
                df = options.get_calls(ticker)
            except:
                print('Failed:', ticker)
                continue
        # rename columns to lower cases
        df.rename(columns={'Contract Name': 'contract_name', 
                            'Last Trade Date': 'last_trade_date',
                            'Stirke': 'stirke',
                            'Last Price': 'last_price',
                            'Bid': 'bid',
                            'Ask': 'ask',
                            'Change': 'change',
                            '% Change': 'pct_change',
                            'Volume': 'volume',
                            'Open Interest': 'open_interest',
                            'Implied Volatility': 'implied_volatility'}, inplace = True)
        
        def float_process(data):
            if data != '-':
                return float(data)
            else:
                return data

        def int_process(data):
            if data != '-':
                return int(data)
            else:
                return data

        df['ticker'] = ticker
        df['expiration_date'] = [row.strip(ticker)[:6] for row in df['contract_name'].values]
        df['expiration_date'] = ['20'+row[:2]+'/'+row[2:4]+'/'+row[4:] for row in df['expiration_date'].values]
        df['bid'] = df['bid'].apply(float_process)
        df['last_price'] = df['last_price'].apply(float_process)
        df['ask'] = df['ask'].apply(float_process)
        df['change'] = df['change'].apply(float_process)
        df['volume'] = df['volume'].apply(int_process)
        df['open_interest'] = df['open_interest'].apply(int_process)
        df['directions']= option_type
        df['date'] = str(dt.date.today() - dt.timedelta(days=1))
        df = df.replace({'-':None})
        df = df.replace({np.nan:None})
        df = df.to_dict('records')
        df = [json.dumps(data) for data in df]
        df= '\n'.join(df)

        # Write to JSON File
        if not os.path.exists('sp500_{}'.format(option_type)):
            os.makedirs('sp500_{}'.format(option_type))
        
        with open('sp500_{}/{}_{}.json'.format(option_type, str(dt.date.today()-dt.timedelta(days=1)), option_type), 'a') as outfile:
            outfile.write(df + '\n')

        # track progress 
        print(i, ticker)




