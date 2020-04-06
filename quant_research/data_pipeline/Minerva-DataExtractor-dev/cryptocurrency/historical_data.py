from yahoofinancials import YahooFinancials
import datetime as dt
import json
import os

from sp500.aws_boto3 import upload_to_aws

class Crypto(object):
    def __init__(self, tickers=None, currency=None):
        self.tickers = tickers
        if not self.tickers:
            self.tickers = ['BCH-USD', 'EOS-USD', 'TRX-USD', 'XRP-USD', 'BTC-USD', 'ETH-USD', 'LTC-USD', 'ADA-USD']
        self.yesterday = str(dt.date.today()-dt.timedelta(days=1))

    def _mkdir(self):
        if not os.path.exists('crypto_data'):
            os.makedirs('crypto_data')
            
    def get_crypto(self):
        yahoo = YahooFinancials(self.tickers)
        data = yahoo.get_historical_price_data(start_date='2000-01-01',end_date=self.yesterday, time_interval='daily')
        
        # make directory and store to csv
        self._mkdir()

        print('\n# Start fetching cryptocurrency data for {} products'.format(len(self.tickers)))
        for ticker in self.tickers:
            for date in range(len(data[ticker]['prices'])):
                df = {}
                df['ticker'] = ticker
                df.update(data[ticker]['prices'][date])
                df['date'] = df['formatted_date']
                df.pop('formatted_date')
                df['currency'] = data[ticker]['currency']
                df['eventsData'] = data[ticker]['eventsData']
                df['timeZoneGMT'] = data[ticker]['timeZone']['gmtOffset']
                df['instrumentType'] = data[ticker]['instrumentType']
                df['firstTradeDate'] = data[ticker]['firstTradeDate']['formatted_date']
                
                
                df = json.dumps(df) + '\n'

                # append data to same json file 
                with open('crypto_data/crypto_historical.json', 'a') as outfile:
                    outfile.write(df)
                
            # track progress
            print('wrote:', ticker)

        print('Finished!')

if __name__ == '__main__':
    Crypto().get_crypto()
    upload_to_aws('crypto_data/crypto_historical.json', 'moyi-minerva', 'crypto_data/crypto_historical.json')