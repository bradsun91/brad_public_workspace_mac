import tushare as ts 
from yahoofinancials import YahooFinancials
import pandas as pd 
import os 
import datetime as dt 
import json
import numpy as np

 # call tushare pro api
pro = ts.pro_api('8ef5ec61cdd848715c57c11d58dd71da1271f76b2420d2bac8aef123')
print('Connect successfully!')

class Index(object):
    def __init__(self):
        self.US_tickers = ['DJI', 'SPX', 'IXIC'] 
        self.CN_tickers = ['000001.SH']
        self.HK_tickers = ['HSI']

    def _mk_dir(self):
        if not os.path.exists('indexes_data'):
            os.makedirs('indexes_data')

    def get_indexes(self):
       
        # get major market indexes

        print('# starting fetch US indexes')
        US = pd.DataFrame()
        for ticker in self.US_tickers:
            us = pro.index_global(ts_code=ticker)
            US = pd.concat([US,us],axis=0,sort=False)

        US['country'] = 'United States'
        US.rename(columns={'ts_code':'index'})

        print('# starting fetch HK indexes')
        HK = pd.DataFrame()
        for ticker in self.HK_tickers:
            hk = pro.index_global(ts_code=ticker)
            HK = pd.concat([HK,hk],axis=0,sort=False)
        HK['country'] = 'Hong Kong'
        HK.rename(columns={'ts_code':'index'})

        print('# starting fetch CN indexes')
        CN = pd.DataFrame()
        for ticker in self.CN_tickers:
            cn= pro.index_daily(ts_code=ticker)
            CN = pd.concat([CN,cn],axis=0,sort=False)
        CN['country'] = 'China'
        CN.rename(columns={'ts_code':'index'})

        # merge into one dataframe and save csv
        self._mk_dir()
        data = pd.concat([US,HK,CN],axis=0,sort=False)
        data = data.replace({np.nan:None})
        data = data.to_dict('records')
        data = [json.dumps(df) for df in data]
        data = '\n'.join(data)

        # write to JSON file
        with open('indexes_data/historical_indexes.json','a') as outfile:
            outfile.write(data)
        # main_df.to_csv('indexes_data/historical_indexes.csv')

        # return main_df

if __name__ == '__main__':
    Index().get_indexes()
