#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: denisdeng
"""

import bs4 as bs
import requests
from pandas_datareader import data
import pandas as pd
import yahoo_fin.stock_info as si

#Scrap tickers from wikipedia and arrange them by sector


html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

soup = bs.BeautifulSoup(html.text, features="lxml")

#creating list for each sector
financial=[]
industrials=[]
healthcare=[]
info_tech=[]
communication=[]
discretionary=[]
materials=[]
realestate=[]
utilities=[]
staple=[]
energy=[]

table = soup.find('table', {'class':'wikitable sortable'})

rows = table.findAll('tr')[1:]

for row in rows:
    ticker = row.findAll('td')[0].text
    indus = row.findAll('td')[3].text

    if indus == 'Industrials':
        industrials.append(ticker[:-1])
    if indus == 'Health Care':
        healthcare.append(ticker[:-1])
    if indus == 'Information Technology':
        info_tech.append(ticker[:-1])
    if indus == 'Communication Services':
        communication.append(ticker[:-1])
    if indus == 'Consumer Discretionary':
        discretionary.append(ticker[:-1])
    if indus == 'Consumer Staples':
        staple.append(ticker[:-1])
    if indus == 'Financials':
        financial.append(ticker[:-1])
    if indus == 'Real Estate':
        realestate.append(ticker[:-1])
    if indus == 'Energy':
        energy.append(ticker[:-1])
    if indus == 'Utilities':
        utilities.append(ticker[:-1])
    if indus == 'Materials':
        materials.append(ticker[:-1])
 
print("To view the detail of the components in each sector, please uncomment the print() in the code and put the name of sector list")

## To check the list of tickers in each sector, simply put the name of the list
'''
print(industrials)
'''


#rank tickers by market capitalization
print("\n")
print("Rank Tickers by Market Cap And Show Top 5 Companies")
print("---------------------------------\n")


data1 = data.get_quote_yahoo(industrials)['marketCap']
a = data1.copy()
a.sort_values(ascending = False, inplace = True)

industrials_5 = a[0:5]
print("Top 5 in Industrials:\n",industrials_5)


data2 = data.get_quote_yahoo(healthcare)['marketCap']
b = data2.copy()
b.sort_values(ascending = False, inplace = True)

healthcare_5 = b[0:5]

print("Top 5 in Healthcare are:\n",healthcare_5)


data3 = data.get_quote_yahoo(info_tech)['marketCap']
c = data3.copy()
c.sort_values(ascending = False, inplace = True)

info_tech_5 = c[0:5]

print("Top 5 in Info Technology are:\n",info_tech_5)




data4 = data.get_quote_yahoo(communication)['marketCap']
d = data4.copy()
d.sort_values(ascending = False, inplace = True)

communication_5 = d[0:5]

print("Top 5 in Communication Services are:\n",communication_5)



data5 = data.get_quote_yahoo(discretionary)['marketCap']
e = data5.copy()
e.sort_values(ascending = False, inplace = True)

discretionary_5 = e[0:5]

print("Top 5 in Consumer Discretionary are:\n",discretionary_5)


data6 = data.get_quote_yahoo(materials)['marketCap']
f = data6.copy()
f.sort_values(ascending = False, inplace = True)

materials_5 = f[0:5]

print("Top 5 in Materials are:\n",materials_5)


data7 = data.get_quote_yahoo(realestate)['marketCap']
g = data7.copy()
g.sort_values(ascending = False, inplace = True)

realestate_5 = g[0:5]

print("Top 5 in Real Estate are:\n",realestate_5)


data8 = data.get_quote_yahoo(utilities)['marketCap']
h = data8.copy()
h.sort_values(ascending = False, inplace = True)

utilities_5 = h[0:5]

print("Top 5 in Utilities are:\n",utilities_5)


data9 = data.get_quote_yahoo(energy)['marketCap']
i = data9.copy()
i.sort_values(ascending = False, inplace = True)

energy_5 = i[0:5]

print("Top 5 in Energy are:\n",energy_5)






##BRK.B should have the largest mkt cap, but the yahoo api does not have data

financial.remove('BRK.B')
data10=data.get_quote_yahoo(financial)['marketCap']
i = data10.copy()
i.sort_values(ascending = False, inplace = True)

fin_5 = i[0:5]

print("Top 5 in Financial except BRK.B are:\n",fin_5)



staple.remove('BF.B') #yahoo api doesnt have data for bf.b, but it not one of top 5, so it doesn't matter

data11=data.get_quote_yahoo(staple)['marketCap']
i = data11.copy()
i.sort_values(ascending = False, inplace = True)

staple_5 = i[0:5]

print("Top 5 in Consumer Staple are:\n",staple_5)




#rank the top5 companies by financial metrics
print("\n")
print("Rank the Top 5 companies by financial metrics")
print("---------------------------------\n")

def takeSecond(rank):
    return rank[1]

indus5=['UPS','UNP','HON','LMT','MMM']
rank = []
for stock in indus5:
    pe = si.get_quote_table(stock)['PE Ratio (TTM)']
    temp=[stock,pe]
    rank.append(temp)

rank.sort(key=takeSecond,reverse=True)
print("Industrial sector top 5 ranked by pe\n",rank)

dis5=['AMZN','HD','NKE','MCD','LOW']
rank = []
for stock in dis5:
    pe = si.get_quote_table(stock)['PE Ratio (TTM)']
    temp=[stock,pe]
    rank.append(temp)

rank.sort(key=takeSecond,reverse=True)
print("Consumer Discretionary sector top 5 ranked by pe\n",rank)

mat5=['LIN','APD','SHW','ECL','NEM']
rank = []
for stock in mat5:
    pe = si.get_quote_table(stock)['PE Ratio (TTM)']
    temp=[stock,pe]
    rank.append(temp)

rank.sort(key=takeSecond,reverse=True)
print("Materials sector top 5 ranked by pe\n",rank)


util5=['NEE','DUK','D','SO','AEP']
rank=[]
for stock in util5:
        sheet=si.get_balance_sheet(stock)
        a=sheet.loc['totalLiab'][1]
        b=sheet.loc['totalStockholderEquity'][1]
        de=a/b
        temp=[stock,de]
        rank.append(temp)
rank.sort(key=takeSecond,reverse=True)
print("Utilities sector top 5 ranked by de\n",rank)

real5=['AMT','PLD','EQIX','CCI','DLR']
rank=[]
for stock in real5:
        sheet=si.get_balance_sheet(stock)
        a=sheet.loc['totalLiab'][1]
        b=sheet.loc['totalStockholderEquity'][1]
        de=a/b
        temp=[stock,de]
        rank.append(temp)
rank.sort(key=takeSecond,reverse=True)
print("Real Estate sector top 5 ranked by de\n",rank)


tech5=['AAPL','MSFT','V','NVDA','MA']
rank=[]
for stock in tech5:
    sheet=si.get_income_statement(stock)
    a=sheet.loc["grossProfit"][1]
    b=sheet.loc["totalRevenue"][1]
    grossmargin=a/b
    temp=[stock,grossmargin]
    rank.append(temp)
rank.sort(key=takeSecond,reverse=True)
print("Information Technology sector top 5 ranked by gross profit margin\n",rank)



health5=['JNJ','UNH','PFE','MRK','ABT']
rank=[]
for stock in health5:
    sheet=si.get_cash_flow(stock)
    a=sheet.loc["totalCashFromOperatingActivities"][1]
    sheet=si.get_balance_sheet(stock)
    b=sheet.loc['totalLiab'][1]
    cashflowcov=a/b
    temp=[stock,cashflowcov]
    rank.append(temp)
rank.sort(key=takeSecond,reverse=True)
print("Healthcare sector top 5 ranked by cash flow coverage raio\n",rank)



com5=['GOOGL','GOOG','FB','DIS','NFLX']
rank=[]
for stock in com5:
    sheet=si.get_income_statement(stock)
    a=sheet.loc["grossProfit"][1]
    b=sheet.loc["totalRevenue"][1]
    grossmargin=a/b
    temp=[stock,grossmargin]
    rank.append(temp)
rank.sort(key=takeSecond,reverse=True)
print("Communication Service sector top 5 ranked by gross profit margin\n",rank)


ener5=['XOM','CVX','COP','KMI','WMB']
rank=[]
for stock in ener5:
    sheet=si.get_income_statement(stock)
    a=sheet.loc["ebit"][0]
    sheet=si.get_balance_sheet(stock)
    b=sheet.loc["totalAssets"][1]
    c=sheet.loc["totalCurrentLiabilities"][1]
    roce=a/(b-c)
    temp=[stock,roce]
    rank.append(temp)
rank.sort(key=takeSecond,reverse=True)
print("Energy sector top 5 ranked by ROCE\n",rank)



fin5=['JPM','BAC','BLK','WFC','C']
rank=[]
for stock in fin5:
    p=si.get_quote_table(stock)['Quote Price']
    bp=si.get_stats(stock)['Value'][47]
    pb = p/float(bp)
    temp=[stock,bp]
    rank.append(temp)
rank.sort(key=takeSecond,reverse=True)
print("Financial sector top 5 ranked by P/B\n",rank)



stap5=['WMT','PG','KO','PEP','COST']
rank=[]
for stock in stap5:
    roe=si.get_stats(stock)['Value'][33]
    temp=[stock,float(roe[:-1])]
    rank.append(temp)
    rank.sort(key=takeSecond,reverse=True)
print("Consumer Staple sector top 5 ranked by ROE\n",rank)

