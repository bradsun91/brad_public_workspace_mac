import tushare as ts
import pandas as pd
import tushare.futures.domestic_cons as qh    #获得合约名字
import numpy as np
import datetime as dt 
import os 

pro = ts.pro_api('8ef5ec61cdd848715c57c11d58dd71da1271f76b2420d2bac8aef123')

class Futures(object):
    def __init__(self):
        self.tickers = qh.FUTURE_CODE
        
    def _mk_dir(self):
        if not os.path.exists('futures_data'):
            os.mkdir('futures_data')

    def get_futures(self):
        # get yesterday
        date = dt.date.today()-dt.timedelta(days=1)
        
        # call tushate api 
        print('# Start Fetching futures data')
        data = pd.DataFrame()

        for exchange in ['DCE', 'SHFE', 'CFFEX', 'CZCE']:
            df = pro.fut_daily(trade_date=date.strftime('%Y%m%d'), exchange=exchange)
            df['ticker'] = [ts_code.split('.')[0] for ts_code in list(df['ts_code'])]
            df['exchange'] = [ts_code.split('.')[1] for ts_code in list(df['ts_code'])]
            df.drop(columns=['ts_code'])

            data = pd.concat([data,df],axis=0,sort=False)
            print('--Saved %s' %exchange)
        print('Saved %s\n' %date)
        
        # make directory and save to csv
        self._mk_dir()
        data.drop(columns=['ts_code'],inplace=True)
        data.set_index('trade_date', inplace=True)
        data.to_csv('futures_data/{}_futures_data.csv'.format(str(date)))
        print('Saved to csv!!')

if __name__ == '__main__':
    Futures().get_futures()