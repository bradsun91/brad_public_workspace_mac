

def market_data(tickers, start, end, col="All"):
	"""
	tickers: 可以是一个，也可以是一个list, e.g. ['AAPL',"TSLA"]
	col: 这里可以是一个column，也可以是一个list的columns。比如想要调用一段时间内market_data关于AAPL的open,close，则：
	aapl_open_close = market_data("AAPL", "2020-01-10","2020-01-23", ["Open", "Close"])
	col默认为"All"，代表调用所有columns。
	start: 开始时间, format example: "2020-01-23"
	end: 结束时间，同上

	"""
	return

def options_data(tickers, strike, start, end, expiration_date, direction, col="All"):
	"""
	col：用法同上;
	tickers: 用法同上；
	Start／end: represents the date when we download the options data after the market closes of that day.格式同上
	"""
	return


def fundamental_data(tickers, start, end, sheet，col="All"):
	"""
	col的用法同上;
	tickers: 用法同上；
	sheet这里主要是调用某个特定的财务报表数据：
	1) balance sheet
	2) income statement
	3) cashflow statement 
	4) key_statistics
	"""
	return

def market_cap(tickers, start, end):
	"""
	对应数据：Hansen每日在进行更新的market cap的数据
	tickers: 用法同上；
	返回某一段时间某个ticker或者某些tickers的market cap
	"""
	return