import datetime as dt
import pandas as pd
import numpy as np
import os
import json
from  save_tickers import save_sp500_tickers
from yahoofinancials import YahooFinancials 
from pandas_datareader import data as web

#financial statements 
class sp500_financial():
    def __init__(self,stmt_type=None,frequency=None,tickers=None):
        self.type = stmt_type
        self.frequency = frequency
        self.tickers = tickers

    # create folders to store data
    def mk_dir(self):
        #if not os.path.exists('sp500_fundamentals_dfs'):
            #os.makedirs('sp500_fundamentals_dfs')

        # create folders for financial statements on dates added
        if self.type in ('balance','cash','income'):
            if not os.path.exists('sp500_fundamentals_dfs/{}'.format(dt.date.today())):
                os.makedirs('sp500_fundamentals_dfs/{}'.format(dt.date.today()))

        #create folders for key_statistics on dates added
        if self.type == 'key_statistics':
            if not os.path.exists('sp500_key_statistics_dfs/{}'.format(dt.date.today())):
                os.makedirs('sp500_key_statistics_dfs/{}'.format(dt.date.today()))

        if self.type == 'market_cap':
            if not os.path.exists('sp500_market_cap_dfs'):
                os.makedirs('sp500_market_cap_dfs')

    # make calls to yahoo
    def call_yahoo(self, ticker):
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

        #set up a main dataframe for all tickers
        main_df = pd.DataFrame()
        print('\n# Start Fetching {} {} for {} stocks'.format(self.type, self.frequency, len(tickers)))
        for ticker in tickers:
            ticker = ticker.replace('.','')
            stmt_df = self.call_yahoo(ticker)

            # detect empty financial statements on ticker level
            if stmt_df == None:
                continue
            stmt_df = stmt_df[ticker]

            # a secondary data from for single ticker
            dfs = pd.DataFrame()

            for j in range(len(stmt_df)):
                df = stmt_df[j]
                k = [k for k in df.keys()][0]
                multi_index = pd.MultiIndex.from_product([[k],df[k].keys()], names=['Date','Stats'])
                df = pd.DataFrame([value for value in df[k].values()],index=multi_index)

                # detect empty financial statements on date level
                if df.empty:
                    continue
                df.columns = ['{}'.format(ticker)]

                # concatenate financial statements of different dates
                dfs = pd.concat([dfs,df],axis=0,sort='False')
                # dropping duplicate index in single financial statement 
                dfs = dfs[~dfs.index.duplicated(keep='first')]

            # concatenate ticker financial statement to a main dataframe
            main_df = pd.concat([main_df,dfs],axis=1, sort='False')
            print(ticker)

        print('Finished!!')

        # create folders and store data
        self.mk_dir()
        main_df.to_csv('sp500_fundamentals_dfs/{}/{}stocks_{}_{}.csv'.format(dt.date.today(),len(tickers), self.frequency, self.type))
        print('Loaded!!\n')
        return main_df
    
    # get key statistics
    def get_key_stats_df(self):
        # initiate tickers
        if self.tickers:
            tickers = self.tickers
        else:
            # with open('sp500tickers.pickle','rb') as f:
                # tickers = pickle.load(f)
            tickers = save_sp500_tickers()

        #set up a main dataframe for all tickers
        main_df = pd.DataFrame()
        print('\n# Start Fetching {} for {} stocks'.format(self.type, len(tickers)))
        for ticker in tickers:
            ticker = ticker.replace('.','')
            df = self.call_yahoo(ticker)
            
            # detect empty financial statements after API call
            if df == None:
                continue
            
            df = df[ticker]
            df = pd.DataFrame(df.values(),index=df.keys())

            # detect empty financial statements on ticker level
            if df.empty:
                continue

            df.columns = ['{}'.format(ticker)]

            # dropping duplicate index in single financial statement
            df = df[~df.index.duplicated(keep='first')]

            # concatenate ticker key statistics to a main dataframe
            main_df = pd.concat([main_df,df],axis=1,sort='False')
            print(ticker)

        print('Finished!!')

        # create folders and store data
        self.mk_dir()
        main_df.to_csv('sp500_key_statistics_dfs/{}/{}stocks_{}.csv'.format(dt.date.today(),len(tickers), self.type))
        print('Loaded!!\n')
        return main_df

    # get ticker_sector dataframe and output csv 
    def get_ticker_sector_df(self):

        print('\n# Start Fetching {} for sp500'.format('tickers and sectors'))

        ticker_sector = save_sp500_tickers(sector=True)
        ticker_sector = pd.DataFrame(ticker_sector,columns=['tickers','sectors'])
        ticker_sector.set_index(['tickers'],inplace=True)

        # get tickers and sectors for required 
        if self.tickers:
            ticker_sector = ticker_sector.loc[self.tickers]

        print('Finished!')

        #output csv file
        ticker_sector.to_csv('sp500_tickers_sectors.csv')
        print('Loaded!')

        return ticker_sector

    # get market capitalizations
    def get_market_cap_df(self):
        # initiate tickers
        if self.tickers:
            tickers = self.tickers
        else:
            tickers = save_sp500_tickers()

        print('\n# Start Fetching {}'.format('market capitalization for sp500'))

        # create a main dataframe to store all market capitals
        main_df = pd.DataFrame()
        
        # list to store failed attempts on tickers (due to no market capital)
        fail_list = []
        
        # iterate each ticker
        for ticker in tickers:
            # track progress
            ticker = ticker.replace('.','')

            try:
                # make call to get market capital
                market_cap = web.get_quote_yahoo(ticker)['marketCap']
                print(ticker)

            except:
                # print failed ticker and append to fail list 
                print('Failed: {}'.format(ticker))
                fail_list.append(ticker)

                # ignore ticker with no market captial
                continue

            # build ticker and marketCap dataframe for single ticker
            market_cap = pd.DataFrame(market_cap)
            market_cap.reset_index(inplace=True)
            market_cap.columns = ['tickers','marketCap']

            # concatenate to main dataframe
            main_df = pd.concat([main_df,market_cap],axis=0,sort=False)
        
        # add date to main dataframe and set as index
        main_df['date'] = dt.date.today()
        main_df.set_index(['date'],inplace=True)

        print('Finished!')

        # output csv file
        main_df.to_csv('sp500_market_cap_dfs/sp500_market_capitalization_{}.csv'.format(dt.date.today()))
        print('Loaded!\n')

        # print failed list
        print('Ticker Failed List:', fail_list)

        return main_df


if __name__ == '__main__':
    # testing tickers
    #tickers = ['MSFT','TSLA','AMZN']

    # initiate SP500 annually financial statement objects
    balance_annual = sp500_financial('balance',frequency='annual')
    cash_annual = sp500_financial('cash',frequency='annual')
    income_annual = sp500_financial('income',frequency='annual')
    
    # initiate sp500 quarterly financial statements objects
    balance_quarter = sp500_financial('balance',frequency='quarterly')
    cash_quarter = sp500_financial('cash',frequency='quarterly')
    income_quarter = sp500_financial('income',frequency='quarterly')

    # initiate sp500 key statistics object
    key = sp500_financial('key_statistics')

    # get financial statements and key statistics 
    # store in csv files and return pandas dataframes
    # keep track of progress

    # call objects and get annual financial statements
    balance_annual_df = balance_annual.get_stmts_df()
    cash_annual_df = cash_annual.get_stmts_df()
    income_annual_df = income_annual.get_stmts_df()

    # call objects and get quarterly financial statements
    balance_quarter_df = balance_quarter.get_stmts_df()
    cash_quarter_df = cash_quarter.get_stmts_df()
    income_quarter_df = income_quarter.get_stmts_df()

    # call objects and get key statistics
    key_df = key.get_key_stats_df()

    # call objects and get tickers and sectors
    '''ticker_sector = sp500_financial('industry_sector')
    ticker_sector_df = ticker_sector.get_ticker_sector_df()'''

    # call objects and get market capitalization
    market_cap = sp500_financial('marekt_cap')
    market_cap_df = market_cap.get_market_cap_df()


