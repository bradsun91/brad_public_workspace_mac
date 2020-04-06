import datetime as dt
import pandas as pd
import numpy as np
import os
import json
from yahoofinancials import YahooFinancials 
from pandas_datareader import data as web

from .save_tickers import save_sp500_tickers


#financial statements 
class sp500_financial():
    def __init__(self,stmt_type=None,frequency=None,tickers=None):
        self.type = stmt_type
        self.frequency = frequency
        self.tickers = tickers

    # create folders to store data
    def _mk_dir(self):
        #if not os.path.exists('sp500_fundamentals_dfs'):
            #os.makedirs('sp500_fundamentals_dfs')

        # create folders for financial statements on dates added
        if self.type in ('balance','cash','income'):
            if not os.path.exists('sp500_fundamentals_dfs'):
                os.makedirs('sp500_fundamentals_dfs')

        #create folders for key_statistics on dates added
        if self.type == 'key_statistics':
            if not os.path.exists('sp500_key_statistics_dfs'):
                os.makedirs('sp500_key_statistics_dfs')

        if self.type == 'market_cap':
            if not os.path.exists('sp500_market_cap_dfs'):
                os.makedirs('sp500_market_cap_dfs')

    # make calls to yahoo
    def _call_yahoo(self, ticker):
        if self.type == 'balance': 
           balance_sheets = YahooFinancials(ticker).get_financial_stmts(self.frequency,self.type)
           if self.frequency == 'quarterly': balance_sheets = balance_sheets['balanceSheetHistoryQuarterly']
           else: balance_sheets = balance_sheets['balanceSheetHistory']
           return balance_sheets
    
        elif self.type == 'cash': 
            cashflow_stmts = YahooFinancials(ticker).get_financial_stmts(self.frequency,self.type)
            if self.frequency == 'quarterly': cashflow_stmts = cashflow_stmts['cashflowStatementHistoryQuarterly']
            else: cashflow_stmts = cashflow_stmts['cashflowStatementHistory']
            return cashflow_stmts

        elif self.type == 'income': 
            income_stmts = YahooFinancials(ticker).get_financial_stmts(self.frequency,self.type)
            if self.frequency == 'quarterly': income_stmts = income_stmts['incomeStatementHistoryQuarterly']
            else: income_stmts = income_stmts['incomeStatementHistory']
            return income_stmts

        elif self.type == 'key_statistics': 
            key_stats = YahooFinancials(ticker).get_key_statistics_data()
            return key_stats

    # get financial statements
    def get_stmts_df(self):
        # initiate tickers
        if self.tickers:
            tickers = self.tickers
        else:
            #with open('sp500tickers.pickle','rb') as f:
                #tickers = pickle.load(f)
            tickers = save_sp500_tickers()

        # create folders and store data
        self._mk_dir()

        # Fetch and clean data 
        print('\n# Start Fetching {} {} for {} stocks'.format(self.type, self.frequency, len(tickers)))
        for ticker in tickers:
            ticker = ticker.replace('.','')

            try:
                stmt_df = self._call_yahoo(ticker)
            except:
                # ignore company with no specified financial statment 
                print('Failed: {}'.format(ticker))
                continue
            
            # access first level dict
            stmt_df = stmt_df[ticker]
            
            # if statement is none, continue
            if stmt_df == None:
                continue

            for j in range(len(stmt_df)):
                # access financial statement on specified date 
                df = stmt_df[j]

                # get date for financial statement
                k = [k for k in df.keys()][0]

                # add ticker and date to dict
                df = df[k]
                df['ticker'] = ticker
                df['date'] = k
                df = json.dumps(df) + '\n'

                # append data to same json file 
                with open('sp500_fundamentals_dfs/{}_{}_{}.json'.format(str(dt.date.today()), self.type, self.frequency), 'a') as outfile:
                    outfile.write(df)

            # track ticker progress
            print('Wrote:',ticker)
        
        print('Finished!!')
        
    # get key statistics
    def get_key_stats_df(self):
        # initiate tickers
        if self.tickers:
            tickers = self.tickers
        else:
            # with open('sp500tickers.pickle','rb') as f:
                # tickers = pickle.load(f)
            tickers = save_sp500_tickers()

       # create folders and store data
        self._mk_dir()

        print('\n# Start Fetching {} for {} stocks'.format(self.type, len(tickers)))
        for ticker in tickers:
            ticker = ticker.replace('.','')
            df = self._call_yahoo(ticker)
            
            # detect empty financial statements after API call
            try:
                df = self._call_yahoo(ticker)
            except:
                # ignore company with no specified financial statment 
                print('Failed: {}'.format(ticker))
                continue
            
            # access key statistics for each ticker
            df = df[ticker]
            if not df:
                print('Failed: {}'.format(ticker))
                continue
                
            df['ticker'] = ticker
            df['date'] = str(dt.date.today()-dt.timedelta(days=1))
            df = json.dumps(df) + '\n'

            # append data to same json file 
            with open('sp500_key_statistics_dfs/{}_{}.json'.format(str(dt.date.today()-dt.timedelta(days=1)), self.type), 'a') as outfile:
                outfile.write(df)

            # keep track progress
            print('Wrote:',ticker)

        print('Finished!!')

    # get market capitalizations
    def get_market_cap_df(self):
        # initiate tickers
        if self.tickers:
            tickers = self.tickers
        else:
            tickers = save_sp500_tickers()

        print('\n# Start Fetching {}'.format('market capitalization for sp500'))
        
        # list to store failed attempts on tickers (due to no market capital)
        fail_list = []
        
        # make directory
        self._mk_dir()

        # iterate each ticker
        for ticker in tickers:
            # track progress
            ticker = ticker.replace('.','')

            try:
                # make call to get market capital
                market_cap = web.get_quote_yahoo(ticker)['marketCap']
            except:
                # print failed ticker and append to fail list 
                print('Failed: {}'.format(ticker))
                fail_list.append(ticker)

                # ignore ticker with no market captial
                continue

            # build ticker and marketCap dataframe for single ticker
            market_cap = {'ticker':ticker,'marketCap':int(float(market_cap[0]))}
            market_cap['date'] = str(dt.date.today()-dt.timedelta(days=1))
            market_cap = json.dumps(market_cap) + '\n'

            # append to JSON file
            with open('sp500_market_cap_dfs/{}_market_cap.json'.format(str(dt.date.today()-dt.timedelta(days=1))), 'a') as outfile:
                outfile.write(market_cap)

            print(ticker)

        print('Finished!')
        print('Loaded!\n')

        # print failed list
        print('Ticker Failed List:', fail_list)