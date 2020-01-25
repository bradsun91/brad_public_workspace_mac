import save_tickers 
import pickle
import os
import pandas as pd
import json
from yahoofinancials import YahooFinancials 

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_tickers.save_sp500_tickers()
    else: 
        with open('sp500tickers.pickle','rb') as f:
            tickers = pickle.load(f)
    
    if not os.path.exists('sp500_fundamentals_dfs'):
        os.makedirs('sp500_fundamentals_dfs')
    
    income_stmts = {}
    cashflow_stmts = {}
    balance_sheets = {}
    key_statistics = {}

    income_stmts['incomeStatementHistoryQuarterly'] = []
    cashflow_stmts['cashflowStatementHistoryQuarterly'] = []
    balance_sheets['balanceSheetHistoryQuarterly'] = []
    key_statistics['keyStatisticsData'] = []
    
    # Alternative way of fetching finanical statements; Can't track progress
    ##income_stmts['incomeStatementHistoryQuarterly'] = [YahooFinancials(ticker).get_financial_stmts('quarterly','income')['incomeStatementHistoryQuarterly'] for ticker in tickers
    ##cashflow_stmts['cashflowStatementHistoryQuarterly'] = [YahooFinancials(ticker).get_financial_stmts('quarterly','cash')['cashflowStatementHistoryQuarterly'] for ticker in tickers
    ##balance_sheets['balanceSheetHistoryQuarterly'] = [YahooFinancials(ticker).get_financial_stmts('quarterly','balance')['balanceSheetHistoryQuarterly'] for ticker in tickers
    ##key_statistics['keyStatisticsData'] = [YahooFinancials(ticker).get_key_statistics_data() for ticker in tickers]

    for ticker in tickers:
        # Fetch financial statements for each ticker
        # For mac users: you might need to install and upgrade Certificates.command using below Terminal command
        # cd /Applications/Python\ 3.7/./Install\ Certificates.command
        all_statements_qt = YahooFinancials(ticker).get_financial_stmts('quarterly',['income','cash','balance'])
        income = all_statements_qt['incomeStatementHistoryQuarterly']
        cash = all_statements_qt['cashflowStatementHistoryQuarterly']
        balance = all_statements_qt['balanceSheetHistoryQuarterly']
        key_statistics_data = YahooFinancials(ticker).get_key_statistics_data()

        # append financial statements from each ticker to designated dicts
        income_stmts['incomeStatementHistoryQuarterly'].append(income)
        cashflow_stmts['cashflowStatementHistoryQuarterly'].append(cash)
        balance_sheets['balanceSheetHistoryQuarterly'].append(balance)
        key_statistics['keyStatisticsData'].append(key_statistics_data)

        # print ticker to check progress
        print(str(ticker))

    # store local json files
    with open('sp500_fundamentals_dfs/sp500_income_statements.json','w+') as outfile:
        json.dump(income_stmts, outfile)
    with open('sp500_fundamentals_dfs/sp500_cashflow_statements.json','w+') as outfile:
        json.dump(cashflow_stmts, outfile)
    with open('sp500_fundamentals_dfs/sp500_balance_sheets.json','w+') as outfile:
        json.dump(balance_sheets, outfile)
    with open('sp500_fundamentals_dfs/sp500_key_statistics.json','w+') as outfile:
        json.dump(key_statistics, outfile)

    return income_stmts,cashflow_stmts,balance_sheets,key_statistics

if __name__ == '__main__':
    income_stmts, cashflow_stmts, balance_sheets, key_statistics_data = get_data_from_yahoo(reload_sp500=False)
    print(income_stmts)