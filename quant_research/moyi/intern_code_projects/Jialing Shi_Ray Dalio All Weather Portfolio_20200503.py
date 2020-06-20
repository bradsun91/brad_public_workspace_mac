#!/usr/bin/env python
# coding: utf-8

# #Ray Dalio All Weather Portfolio

# #Without rebalance

# In[12]:


import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import quantstats as qt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt


# In[15]:


#Get data from yahoo finance
VTI = pdr.get_data_yahoo('VTI', start = '2007-01-11', end = dt.date.today())
TLT = pdr.get_data_yahoo('TLT', start = '2007-01-11', end = dt.date.today())
IEI = pdr.get_data_yahoo('IEI', start = '2007-01-11', end = dt.date.today())
GLD = pdr.get_data_yahoo('GLD', start = '2007-01-11', end = dt.date.today())
GSG = pdr.get_data_yahoo('GSG', start = '2007-01-11', end = dt.date.today())


# In[3]:


#Weight different ETFs
peso_VTI = 0.3
peso_TLT = 0.4
peso_IEI = 0.15
peso_GLD = 0.075
peso_GSG = 0.075


# In[16]:


#Get Adjusted closing price of different ETFs
assets = pd.DataFrame()
assets['VTI'] = VTI['Adj Close'].pct_change()
assets['TLT'] = TLT['Adj Close'].pct_change()
assets['IEI'] = IEI['Adj Close'].pct_change()
assets['GLD'] = GLD['Adj Close'].pct_change()
assets['GSG'] = GSG['Adj Close'].pct_change()
assets = assets.dropna()


# In[17]:


#Apply Ray Dalio All Weather portfolio
all_weather = peso_VTI * assets['VTI'] + peso_TLT * assets['TLT'] + peso_IEI * assets['IEI'] + peso_GLD * assets['GLD'] + peso_GSG * assets['GSG']


# In[6]:


qt.reports.full(all_weather,"VTI")


# In[20]:


qt.reports.full(all_weather,"TLT")


# In[21]:


qt.reports.full(all_weather,"IEI")


# In[22]:


qt.reports.full(all_weather,"GLD")


# In[23]:


qt.reports.full(all_weather,"GSG")


# In[31]:


qt.reports.full(all_weather)


# In[1]:


#Calculate expected returns when investment = 1,000,000
#Cumulative Return of Strategy is 150.36%
initial_investment = 1000000
returns_expected = initial_investment * 1.5036
returns_expected  #1503600.0


# #With rebalance

# In[42]:


class BasicTemplateAlgorithm:
    '''Basic template algorithm simply initializes the date range and cash'''

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''
        self.SetStartDate(2001,1,11)  #Set Start Date
        self.SetEndDate(dt.date.today())    #Set End Date
        self.SetCash(1000000)           #Set Strategy Cash
        
        # Dividend Handling
        self.raw_handling = True
        
        # Simulate topping up your account with savings every period 
        self.savings_on = False
        self.savings_amt = 1000
        
        # This is to stop us adding savings on the first rebalance as it is 
        # immediately after starting the algo
        self.first_rebalance = True
        
        
        # This dictionary will be looped through to add equities and setholdings 
        # It can be expanded to hold more ETF's/Equities.
        self.all_weather = {
            "Equity":{
                    "Ticker": "VTI", #Vanguard Total Stock Market ETF
                    "Weight": 0.3,
                    },
            "Bonds Long-Term":{
                    "Ticker": "TLT", #iShares 20+ Year Treasury Bond ETF
                    "Weight": 0.4,
                    },        
            
            "Bonds Med-Term":{
                    "Ticker": "IEI", #iShares 3-7 Year Treasury Bond ETF
                    "Weight": 0.15,
                    },
           
            "Commodity 1":{
                    "Ticker": "GLD", #SPDR Gold Trust
                    "Weight": 0.075,
                    },
            "Commodity 2":{
                    "Ticker": "GSG", #iShares S&P GSCI Commodity Indexed Trust
                    "Weight": 0.075,
                    },
                    
            }
            
        
        # Setup IB Broker simulation
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
            
        
        # Add The ETF'S!
        # ---------------
        for key, asset in self.all_weather.items():
            self.AddEquity(asset["Ticker"], Resolution.Daily)
            
            # Set Dividend Handling Method
            # ----------------------------
            # https://www.quantconnect.com/forum/discussion/508/update-dividends-splits-and-custom-price-normalization/p1
            if self.raw_handling:
                self.Securities[asset["Ticker"]].SetDataNormalizationMode(DataNormalizationMode.Raw)
            else:
                self.Securities[asset["Ticker"]].SetDataNormalizationMode(DataNormalizationMode.TotalReturn)
        
        
        # We will assume that if we can place an order for the Equity, then the other
        # ETF's should be fine. 
        self.Schedule.On(self.DateRules.MonthStart(self.all_weather["Equity"]["Ticker"]),
                            self.TimeRules.AfterMarketOpen(self.all_weather["Equity"]["Ticker"]),
                            self.Rebalance)
                            

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.

        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''
        # Log any dividends received.
        # ---------------------------
        for kvp in data.Dividends: # update this to Dividends dictionary
            div_ticker = kvp.Key
            div_distribution = kvp.Value.Distribution
            div_total_value = div_distribution * self.Portfolio[div_ticker].Quantity
            self.Log("DIVIDEND >> {0} - ${1} - ${2}".format(div_ticker, div_distribution, div_total_value))
            
            
    def Rebalance(self):
        month = self.Time.month
        
        # Return if we don't want to rebalance this month
        # Add extra months in here to rebalance more often
        # i.e for March insert 3 into the list. 
        if month not in [1,6]: return
    
        self.Log('-------------------->>')
        self.Log("{0} RE-BALANCE >> Total Value {1} | Cash {2}".format(
                                                                    self.Time.strftime('%B').upper(),
                                                                    self.Portfolio.TotalPortfolioValue,
                                                                    self.Portfolio.Cash))
    
    
        if self.savings_on and not self.first_rebalance:
            
            cash_after_savings = self.Portfolio.Cash + self.savings_amt
            self.Log("Top Up Savings >> New Cash Balance {0}".format(
                                                                cash_after_savings))
            self.Portfolio.SetCash(cash_after_savings)
    
        # Rebalance!                                                                
        for key, asset in self.all_weather.items():
            
            holdings = self.Portfolio[asset["Ticker"]].Quantity
            price = self.Portfolio[asset["Ticker"]].Price
            
            self.Log("{0} >> Current Holdings {1} | Current Price {2}".format(
                                                                    self.Portfolio[asset["Ticker"]].Symbol,
                                                                    holdings,
                                                                    price))

            self.SetHoldings(asset["Ticker"], asset["Weight"])
            
        self.Log('-------------------->>')
        
        # Set first rebalance to False so we add the savings next time around
        # (if turned on)
        self.first_rebalance = False

