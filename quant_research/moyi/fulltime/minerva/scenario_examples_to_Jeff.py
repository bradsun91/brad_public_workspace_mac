

def market_data(tickers, start, end, col="All"):
	"""
	tickers: 可以是一个，也可以是一个list, e.g. ['AAPL',"TSLA"]
	col: 这里可以是一个column，也可以是一个list的columns。比如想要调用一段时间内market_data关于AAPL的open,close，则：
	aapl_open_close = market_data("AAPL", "2020-01-10","2020-01-23", ["Open", "Close"])
	col默认为"All"，代表调用所有columns。
	start: 开始时间, format example: "2020-01-23"
	end: 结束时间，同上

	1. tickers 参数是不是必须的？如果不是，是不是返回所有ticker？
	- Brad: 参数不是必须；如果不是，返回所有ticker对应的数据（不仅仅是ticker）


	2. start, end 这对时间参数是不是必须的？如果不是，是不是返回所有时间段的data?
	- Brad: 不是必须；如果不是，返回所有时间段的数据


	3. 时间段过大或者ticker太多时，返回数据量需不需要做分页
	- Brad: 按照Jeff的建议来做


	4. 返回的数据如何排序
	- Brad: 按照时间的顺序，最新的数据放在最下面，最旧的数据放在最上面

	"""
	return

def options_data(tickers, strike, start, end, expiration_date, direction, col="All"):
	"""
	col：用法同上;
	tickers: 用法同上；
	Start／end: represents the date when we download the options data after the market closes of that day.格式同上


	1. tickers 参数是不是必须的？如果不是，是不是返回所有ticker？
	- Brad: 参数不是必须；如果不是，返回所有ticker对应的数据（不仅仅是ticker）


	2. start, end 这对时间参数是不是必须的？如果不是，是不是返回所有时间段的data?
	- Brad: 不是必须；如果不是，返回所有时间段的数据


	3. 时间段过大或者ticker太多时，返回数据量需不需要做分页
	- Brad: 按照Jeff的建议来做


	4. 返回的数据如何排序
	- Brad: 按照时间的顺序，最新的数据放在最下面，最旧的数据放在最上面

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

	1. tickers 参数是不是必须的？如果不是，是不是返回所有ticker？
	- Brad: 参数不是必须；如果不是，返回所有ticker对应的数据（不仅仅是ticker）


	2. start, end 这对时间参数是不是必须的？如果不是，是不是返回所有时间段的data?
	- Brad: 不是必须；如果不是，返回所有时间段的数据


	3. 时间段过大或者ticker太多时，返回数据量需不需要做分页
	- Brad: 按照Jeff的建议来做


	4. 返回的数据如何排序
	- Brad: 按照时间的顺序，最新的数据放在最下面，最旧的数据放在最上面

	"""
	return

def market_cap(tickers, start, end):
	"""
	对应数据：Hansen每日在进行更新的market cap的数据
	tickers: 用法同上；
	返回某一段时间某个ticker或者某些tickers的market cap

	1. tickers 参数是不是必须的？如果不是，是不是返回所有ticker？
	- Brad: 参数不是必须；如果不是，返回所有ticker对应的数据（不仅仅是ticker）


	2. start, end 这对时间参数是不是必须的？如果不是，是不是返回所有时间段的data?
	- Brad: 不是必须；如果不是，返回所有时间段的数据


	3. 时间段过大或者ticker太多时，返回数据量需不需要做分页
	- Brad: 按照Jeff的建议来做


	4. 返回的数据如何排序
	- Brad: 按照时间的顺序，最新的数据放在最下面，最旧的数据放在最上面


	"""
	return

# ==========  20200215 new added api  ===========
def news_data(tickers, start, end):
	"""
	1. tickers 参数是不是必须的？如果不是，是不是返回所有ticker？
	- Brad: 参数不是必须；如果不是，返回所有ticker对应的数据（不仅仅是ticker）


	2. start, end 这对时间参数是不是必须的？如果不是，是不是返回所有时间段的data?
	- Brad: 不是必须；如果不是，返回所有时间段的数据


	3. 时间段过大或者ticker太多时，返回数据量需不需要做分页
	- Brad: 按照Jeff的建议来做


	4. 返回的数据如何排序
	- Brad: 按照时间的顺序，最新的数据放在最下面，最旧的数据放在最上面
	"""
	return

def macroeconomics_data(indicator, start, end):

		"""
	1. indicator 参数是不是必须的？如果不是，是不是返回所有indicator？
	- Brad: 参数不是必须；如果不是，返回所有ticker对应的数据（不仅仅是indicator）


	2. start, end 这对时间参数是不是必须的？如果不是，是不是返回所有时间段的data?
	- Brad: 不是必须；如果不是，返回所有时间段的数据


	3. 时间段过大或者indicator太多时，返回数据量需不需要做分页
	- Brad: 按照Jeff的建议来做


	4. 返回的数据如何排序
	- Brad: 按照时间的顺序，最新的数据放在最下面，最旧的数据放在最上面
	"""
	return
