# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 05:15:28 2020

@author: ywu37
"""
import numpy as np
import pandas as pd
import yfinance as yf
#import matplotlib.pyplot as plt
#from bs4 import BeautifulSoup
#import re
from fredapi import Fred
import time
from finsymbols import symbols
#from stock_symbols import symbols
#Import pandas_datareader as pdr
#pdr.get_nasdaq_symbols()
#NYSE = symbols.get_nyse_symbols()
#NASDAQ = symbols.get_nasdaq_symbols()
#AMEX = symbols.get_amex_symbols()
#NYSE_list = pd.read_csv("C:/Users/ywu37/Desktop/MoyiQuant 实习/任务/Fear&Greed Index/NYSE.csv")
#NYSE = list(NYSE_list['Symbol'])
Fred_API_key = 'a40841f52075d4615cb0d3c895819d1c'

def CalculateIndex(series):
    MAX = max(series)
    MIN = min(series)
    index = 100*((series[-1]-MAX)/(MAX-MIN))
    return index

def stock_price_momentum():
    spx = yf.Ticker("^GSPC")
    hist = spx.history(period = "2y")
    hist_close = hist['Close']
    hist_momentum = []
    for i in range(len(hist)-125):
        a = hist_close[125+i]/hist_close[0+i:125+i].mean()-1
        hist_momentum.append(a)
    #index = (hist_momentum[-1]-min(hist_momentum))/(max( hist_momentum)-min( hist_momentum))
    index = CalculateIndex(hist_momentum)
    return index

def market_volatility():
    spx = yf.Ticker("^VIX")
    hist = spx.history(period = "1y")
    hist_close = hist['Close']
    #pct = (hist_close[-1]-min(hist_close))/(max(hist_close)-min(hist_close))
    index = CalculateIndex(hist_close)
    return index

def get_stock_info(ticker, p = '1y'):
    ticker = yf.Ticker(ticker)
    hist = ticker.history(period = p)
    return hist

def stock_price_strength(tickers):
    noh = 0 #number of hitting high
    nol = 0 #number of hitting low
    for i in range(len(tickers)):
        try:
            hist = get_stock_info(tickers[i])
            close = hist['Close']
            if close[-1] == min(close):
                nol += 1
            if close[-1] == max(close):
                noh += 1
        except:
            continue
        
    index = abs(noh - nol)/min([noh, nol])*100
    return index
        
def stock_price_breadth(tickers):
    length = len(get_stock_info('MSFT','1mo'))
    Rvolume = [0 for i in range(length)] #volume of rise during 1 mo
    Dvolume = [0 for i in range(length)] #volume of decline during 1 mo
    for i in range(len(tickers)):
        try:
            hist = get_stock_info(tickers[i], '1mo')
            for j in range(len(hist)):
                if hist['Open'][j] > hist['Close'][j]:
                    Rvolume[j] += hist['Volume'][j]
                else:
                    Dvolume[j] += hist['Volume'][j]
        except:
            continue
        
    breadth = [Rvolume[i] - Dvolume[i] for i in range(len(Rvolume))]
    breadth = [ i for i in breadth if i == i]
    index = CalculateIndex(breadth)
    return index
    
def junk_bond_demand(API):
    date_end = time.strftime('%Y.%m.%d',time.localtime(time.time()))#today date
    date_start = time.strftime('%Y.%m.%d',time.localtime(time.time()-2592000))#date of one month ago
    fred = Fred(api_key=API)
    junkbond = fred.get_series('BAMLH0A0HYM2EY',date_start,date_end) #junkbond data
    investbond = fred.get_series('DAAA',date_start,date_end) #investment-grade bond
    spread = [junkbond [i]-investbond[i] for i in range(min([len(investbond),len(junkbond)]))]
    spread = [ i for i in spread if i == i]#remove nan value
    #index = (spread[-1] - min(spread))/(max(spread)-min(spread))
    index = CalculateIndex(spread)
    return index

def safe_heaven_demand():
    spx = yf.Ticker("^GSPC")
    hist = spx.history(period = "1y")
    hist_close = hist['Close']
    date_end = time.strftime('%Y.%m.%d',time.localtime(time.time()))#today date
    date_start = time.strftime('%Y.%m.%d',time.localtime(time.time()-2592000))#date of one month ago
    fred = Fred(api_key=Fred_API_key)
    treasury = fred.get_series('DGS10',date_start,date_end)# yield of treasuty
    Returns = np.diff(hist_close) / hist_close[:-1] # the return of SP 500
    returns = Returns[-len(treasury):]
    spread = [returns[i] - treasury[i] for i in range(len(returns))]
    spread = [ i for i in spread if i == i]#remove nan value
    #index = (spread[-1] - min(spread))/(max(spread)-min(spread))
    index = CalculateIndex(spread)
    return index

def get_date_list(begin_date,end_date):

    date_list = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start=begin_date, end=end_date))]

    return date_list

#print(get_date_list('2018-06-01','2018-06-08'))

def Put_and_Call_Options(tickers):
    date_end = time.strftime('%Y.%m.%d',time.localtime(time.time()))#today date
    date_start = time.strftime('%Y.%m.%d',time.localtime(time.time()-2592000))#date of one month ago
    date_list = get_date_list(date_start, date_end)
    Put_volume = [0 for i in range(len(date_list))]
    Call_volume = [0 for i in range(len(date_list))]
    for i in range(len(tickers)):
        try:
            stock = yf.Ticker(tickers[i])
            opt = stock.option_chain(stock.options[0])
            callvolume = opt.calls['volume']
            calldate = [x.strftime('%Y-%m-%d') for x in list(opt.calls['lastTradeDate'])]
            putvolume = opt.puts['volume']
            putdate = [x.strftime('%Y-%m-%d') for x in list(opt.puts['lastTradeDate'])]
            for j in range(len(date_list)):
                for k in range(len(calldate)):
                    if date_list[j] == calldate[k]:
                        Call_volume[j] += callvolume[k]
                for k in range(len(putdate)):
                    if date_list[j] == putdate[k]:
                        Put_volume[j] += putvolume[k]                   
        except:
            continue
        
    volumespread = [Call_volume[i] - Put_volume[i] for i in range(len(date_list))]
    volumespread = [ i for i in volumespread if i == i]
    index = CalculateIndex(volumespread)
    return index

def main():
    #NYSE_list = pd.read_csv("C:/Users/ywu37/Desktop/MoyiQuant 实习/任务/Fear&Greed Index/NYSE.csv")
    #NYSE = list(NYSE_list['Symbol'])
    NYSE = symbols.get_nyse_symbols()
    Fred_API_key = 'a40841f52075d4615cb0d3c895819d1c'
    a = Put_and_Call_Options(NYSE)
    b = safe_heaven_demand()
    c = stock_price_breadth(NYSE)
    d = stock_price_momentum()
    e = market_volatility()
    f = stock_price_strength(NYSE)
    g = junk_bond_demand(Fred_API_key)
    index = (a+b+c+d+e+f+g)/7
    print(index)     


if __name__ == "__main__":
    main()