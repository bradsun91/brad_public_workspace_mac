#!/usr/bin/env python
# coding: utf-8

# #Historical Simulation Method for Facebook

# In[1]:


#Step 1: Import the necessary libraries
#Data manipulation
import numpy as np


# In[2]:


import pandas as pd


# In[3]:


#Plotting
import matplotlib.pyplot as plt


# In[4]:


import seaborn


# In[5]:


#Data fetching
import yfinance as yf


# In[7]:


#Print tabular data
from tabulate import tabulate


# In[9]:


#Step 2: Import the daily data of Facebooks' stock from Yahoo finance and calculate the daily returns
df = yf.download('FB', '2012-01-01','2018-01-31')


# In[10]:


df['returns'] = df.Close.pct_change()


# In[11]:


df = df.dropna()


# In[13]:


plt.hist(df.returns, bins=40)
plt.xlabel('Returns')
plt.ylabel('Fequency')
plt.grid(True)
plt.show()


# In[14]:


#Step 3: Sort the returns from the lowest to the highest
df.sort_values('returns', inplace = True, ascending = True)


# In[20]:


#Step 4: Calculate the VaR for 90%, 95%, and 99% confidence intervals using quantile function
VaR_90 = df['returns'].quantile(0.1)
VaR_95 = df['returns'].quantile(0.05)
VaR_99 = df['returns'].quantile(0.01)


# In[25]:


print(tabulate([['90%', VaR_90],['95%', VaR_95],['99%', VaR_99]],headers = ['Confidence interval', 'Value at Risk']))


# #The Variance-Covariance Method (Parametric Method) for Facebook

# In[26]:


#Step 1: Import the necessary libraries
#Data manipulation
import numpy as np
import pandas as pd


# In[27]:


#plotting
import matplotlib.pyplot as plt
import seaborn
import matplotlib.mlab as mlab


# In[28]:


#Statistical calculation
from scipy.stats import norm


# In[29]:


#Data fetching
import yfinance as yf


# In[30]:


#Print tabular data
from tabulate import tabulate


# In[31]:


#Step 2: Import the daily data of Facebooks' stock from Yahoo finance and calculate the daily returns
df = yf.download('FB','2012-01-01','2018-01-31')


# In[32]:


df = df[['Close']]


# In[33]:


df['returns'] = df.Close.pct_change()


# In[34]:


#Step 3: Determine the mean and standard deviation of the daily returns. Plot the normal curve against the daily returns
mean = np.mean(df['returns'])


# In[35]:


std_dev = np.std(df['returns'])


# In[38]:


df['returns'].hist(bins = 40, density = True, histtype = 'stepfilled', alpha = 0.5)
x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
plt.plot(x, norm.pdf(x, mean, std_dev), "r")
plt.show()


# In[39]:


#Step 4: Calculate the VaR using piont percentile function
VaR_90 = norm.ppf(1-0.9, mean, std_dev)
VaR_95 = norm.ppf(1-0.95, mean, std_dev)
VaR_99 = norm.ppf(1-0.99, mean, std_dev)


# In[40]:


print(tabulate([['90%', VaR_90],['95%', VaR_95],['99%', VaR_99]],headers = ['Confidence interval', 'Value at Risk']))


# #Monte Carlo Simulation for AAPL

# In[41]:


#Step 1: Import the necessary libraries
import numpy as np
import pandas as pd


# In[43]:


import quandl


# In[44]:


import matplotlib.pyplot as plt


# In[48]:


#Step 2: Import the daily data of AAPL from quandl and calculate returns
quandl.ApiConfig.api_key = '-3SjK5MsLnRy81trD7Aq'


# In[50]:


symbols = ["WIKI/AAPL.4"]


# In[71]:


data = quandl.get(symbols, start_date = "2015-12-31", end_data = "2017-10-31", collapse = "daily")


# In[94]:


#Step 3: Simulate the price
rets = data.pct_change()
rets = rets[1:]


# In[95]:


daily_vol = rets.std()
daily_ret = rets.mean()


# In[96]:


simulation_df = pd.DataFrame()


# In[97]:


num_simulations = 10000
predicted_days = 252


# In[99]:


last_price = data.iloc[-1:,0]
last_price = last_price[0]


# In[100]:


#Created Each Simulation as a Column in df
for x in range(num_simulations):
    count = 0
    
    price_series = []
    price_series.append(last_price)
    
    #Series for Predicted Days
    for i in range(predicted_days):
        if count == 251:
            break
            
        #price = price_series[count]+price_series[count]*(daily_ret+daily_vol*np.random.normal(0,1))
        price = price_series[count]*(1 + np.random.normal(0, daily_vol))
            
        price_series.append(price)
        count += 1
            
    simulation_df[x] = price_series
    
plt.plot(simulation_df)


# In[104]:


#Step 4: Calculate VaR at 90%, 95%, and 99% confidence intervals
price_array = simulation_df.iloc[-1,:]
price_array = sorted(price_array, key = int)


# In[105]:


var_95 = np.percentile(price_array, 0.95)
var_99 = np.percentile(price_array, 0.99)
var_9999 = np.percentile(price_array, 0.9999)


# In[106]:


print("VaR at 95% Confidence:" + "${:,.2f}".format(last_price - var_95))
print("VaR at 99% Confidence:" + "${:,.2f}".format(last_price - var_99))
print("VaR at 99.99% Confidence:" + "${:,.2f}".format(last_price - var_9999))


# In[109]:


plt.hist(price_array, density = True)
plt.xlabel('Price')
plt.ylabel('Probability')
plt.title(r'Histogram of Simulated Stock Prices', fontsize = 18, fontweight = 'bold')
plt.axvline(x = var_95, color = 'r', linestyle = '--', label = 'Price at Confidence Interval:' + str(round(var_95,2)))
plt.axvline(x = last_price, color = 'k', linestyle = '--', label = 'Current Stock Price:' + str(round(last_price,2)))
plt.legend(loc = 'upper right', fontsize = 'x-small')
plt.show()


# #Case: VaR of a stock portfolio

# In[111]:


#Step 1: Import the necessary libraries
import pandas as pd
from pandas_datareader import data as pdr


# In[112]:


import yfinance as yf


# In[113]:


import numpy as np


# In[114]:


import datetime as dt


# In[121]:


import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# In[123]:


from scipy.stats import norm


# In[115]:


#Step 2: Import the daily data of AAPL, FB, C, and DIS from Yahoo Finance to get the portfolio and calculate periodic returns
#Create the portfolio of equities
tickers = ['AAPL', 'FB', 'C', 'DIS']


# In[116]:


#Set the investment weights
weights = np.array([0.25, 0.3, 0.15, 0.3])


# In[117]:


#Set an initial investment level
initial_investment = 1000000


# In[118]:


#Download closing prices
data = pdr.get_data_yahoo(tickers, start = '2018-01-01', end = dt.date.today())['Close']


# In[119]:


#From the closing prices, calculate periodic returns
returns = data.pct_change()


# In[120]:


returns.tail()


# In[125]:


#Step 3: Create a covariance matrix based on the returns and Calculate the portfolio's mean and standard deviation
#Generate Var-Cov matrix
cov_matrix = returns.cov()
cov_matrix


# In[129]:


#Calculate mean returns for each stock
avg_rets = returns.mean()
avg_rets


# In[128]:


#Calculate mean and standard deviation of returns for portfolio overall
#Using dot product to normalize individual means against investment weights
port_mean = avg_rets.dot(weights)


# In[130]:


port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))


# In[132]:


#Calculate mean and standard deviation of investment
mean_investment = (1 + port_mean)*initial_investment
stdev_investment = initial_investment*port_stdev


# In[133]:


#Step 4: Calculate VaR at 95% confidence interval
#Calculate the inverse of the normal cumulative distribution(PPF) at 95% confidence interval, standard deviation and mean
conf_level1 = 0.05


# In[134]:


#Using Scipy ppf method to generate values for the inverse cumulative distribution function to a normal distribution, plugging in the mean, standard deviation of our portfolio
cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)


# In[136]:


#Subtract the initial investment from the calculation above to estimate the VaR for the portfolio
var_1d1 = initial_investment - cutoff1
var_1d1

#Here we are saying that the portfolio of $1000,000 will not exceed losses greater than $30.4k over a ond day perion at 95% confidence interval


# In[137]:


#Step 5: VaR over n-day time period
#Calculate n Day VaR (1-day to 15-day)
var_array = []


# In[138]:


num_days = int(15)


# In[139]:


for x in range(1, num_days+1):
    var_array.append(np.round(var_1d1*np.sqrt(x),2))
    print(str(x) + "day VaR @ 95% confidence:" + str(np.round(var_1d1*np.sqrt(x),2)))


# In[141]:


#Bulid plot
plt.xlabel("Day #")
plt.ylabel("Max portfolio loss(USD)")
plt.title("Max portfolio loss(VaR) over 15-day period")
plt.plot(var_array, "r");


# In[146]:


#Step 6: Check distributions of four equities against normal distribution
#Repeat for each equity in portfolio
#For AAPL
returns['AAPL'].hist(bins = 40, density =  True, histtype = 'stepfilled', alpha = 0.5)
x = np.linspace(port_mean - 3*port_stdev, port_mean + 3*port_stdev, 100)
plt.plot(x, norm.pdf(x, port_mean, port_stdev), "r")
plt.title("AAPL returns vs. normal distribution");


# In[147]:


#For FB
returns['FB'].hist(bins = 40, density =  True, histtype = 'stepfilled', alpha = 0.5)
x = np.linspace(port_mean - 3*port_stdev, port_mean + 3*port_stdev, 100)
plt.plot(x, norm.pdf(x, port_mean, port_stdev), "r")
plt.title("FB returns vs. normal distribution");


# In[148]:


#For C
returns['C'].hist(bins = 40, density =  True, histtype = 'stepfilled', alpha = 0.5)
x = np.linspace(port_mean - 3*port_stdev, port_mean + 3*port_stdev, 100)
plt.plot(x, norm.pdf(x, port_mean, port_stdev), "r")
plt.title("C returns vs. normal distribution");


# In[149]:


#For DIS
returns['DIS'].hist(bins = 40, density =  True, histtype = 'stepfilled', alpha = 0.5)
x = np.linspace(port_mean - 3*port_stdev, port_mean + 3*port_stdev, 100)
plt.plot(x, norm.pdf(x, port_mean, port_stdev), "r")
plt.title("DIS returns vs. normal distribution");


# In[ ]:




