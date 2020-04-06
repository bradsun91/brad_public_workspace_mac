import tushare as ts
import pandas as pd
import tushare.futures.domestic_cons as qh    #获得合约名字
import numpy as np
import datetime as dt 
import os 
import csv

pro = ts.pro_api('8ef5ec61cdd848715c57c11d58dd71da1271f76b2420d2bac8aef123')

class Futures(object):
    def __init__(self):
        self.tickers = qh.FUTURE_CODE
        
    def _mk_dir(self):
        if not os.path.exists('futures_data'):
            os.mkdir('futures_data')

    def get_futures(self):
        # get historical futures contracts
        datelist = pd.bdate_range(end = pd.datetime.today()-dt.timedelta(days=1), periods = 2000).tolist()
        datelist = [date.strftime('%Y%m%d') for date in datelist]
        
        # make directory 
        self._mk_dir()

        # write up headers
        # headers = ['ts_code','trade_date','pre_close','pre_settle','open','high','low','close',
        #             'settle','change1','change2','vol','amount','oi','oi_chg','ticker','exchange']
        # with open('futures_data/historical_futures.csv', 'a') as f:
        #     csv.writer(f).writerow(headers)

        print('# Start Fetching futures data')
        for date in datelist[datelist.index('20180925'):]:
            for exchange in ['DCE', 'SHFE', 'CFFEX', 'CZCE']:
                df = pro.fut_daily(trade_date=date, exchange=exchange)
                df['ticker'] = [ts_code.split('.')[0] for ts_code in list(df['ts_code'])]
                df['exchange'] = [ts_code.split('.')[1] for ts_code in list(df['ts_code'])]
                df.drop(columns=['ts_code'])

                # save to csv
                with open('futures_data/historical_futures.csv', 'a') as f:
                    csv.writer(f).writerows(df.values)
                # data = pd.concat([data,df],axis=0,sort=False)
                print('--Saved %s' %exchange)
            print('Saved %s\n' %date)
        
        print('Save all contracts!!')
if __name__ == '__main__':
    Futures().get_futures()