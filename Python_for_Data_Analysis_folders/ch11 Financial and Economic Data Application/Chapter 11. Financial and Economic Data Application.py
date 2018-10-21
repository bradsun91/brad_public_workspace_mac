# Chapter 11. Financial and Economic Data Application:

# 1. Time Index Alignment (series 1 has longer timeframe while series 2 has shorter)
Series1.align(series2, join = ‘inner’)

# 2. Deal with different time series frequencies, especially economic data

Ts = Series(series, index = pd.date_range (start_date, periods = 3, freq = ‘W-WED’))
Ts.resample(‘W’)

# Add two time series values with different frequencies together using reindex and ffill:

3. dates
