
import requests
import time
import csv


class Spider(object):
	
	def stmp2date(self, stmp):  
		stmp = float(str(stmp)[:10])
		timeArray = time.localtime(stmp)
		otherStyleTime = time.strftime("%Y-%m-%d", timeArray)
		return otherStyleTime
	
	def get_avg(self, s):
		s = [float(x) for x in s]
		avg = sum(s) / len(s)
		return '%.7f' % avg
	
	def get_rate(self, s1, s2):
		s1 = float(s1)
		s2 = float(s2)
		rate = s1 / s2
		return '%.6f' % rate
	
	def get_detail(self, start, end):
		url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
		
		querystring = {"symbol": "^VIX",
		               "period1": start,
		               "period2": end,
		               "interval": "1d",
		               "includePrePost": "true",
		               "events": "div|split|earn",
		               "lang": "en-US",
		               "region": "US",
		               "crumb": "mAui1KjTQAA",
		               "corsDomain": "finance.yahoo.com"}
		text = requests.get(url, params=querystring, timeout=10).json()
		adjclose_list = text['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']
		close_list = text['chart']['result'][0]['indicators']['quote'][0]['close']
		high_list = text['chart']['result'][0]['indicators']['quote'][0]['high']
		low_list = text['chart']['result'][0]['indicators']['quote'][0]['low']
		open_list = text['chart']['result'][0]['indicators']['quote'][0]['open']
		date_list = text['chart']['result'][0]['timestamp']
		results1 = []
		for i in range(len(date_list)):
			date_str = self.stmp2date(date_list[i])
			open1 = '%.2f' % open_list[i]
			high = '%.2f' % high_list[i]
			low = '%.2f' % low_list[i]
			close = '%.2f' % close_list[i]
			vix = close
			tmp = [date_str, open1, high, low, close, vix]
			print '|'.join(tmp)
			results1.append([x.encode('gbk') for x in tmp])
		with open('result1.csv', 'w') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerows(results1)
		


if __name__ == "__main__":
	spider = Spider()
	end = '1591409112'
	start = '1559786712'
	spider.get_detail(start, end)
	
	