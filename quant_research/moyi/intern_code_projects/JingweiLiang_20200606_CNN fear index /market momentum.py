# -*- coding: utf-8 -*-
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
		url = "https://query2.finance.yahoo.com/v8/finance/chart/%5EGSPC"
		
		querystring = {"formatted": "true",
		               "crumb": "mAui1KjTQAA",
		               "lang": "en-US",
		               "region": "US",
		               "interval": "1d",
		               "period1": start,
		               "period2": end,
		               "events": "div|split",
		               "corsDomain": "finance.yahoo.com"}
		text = requests.get(url, params=querystring, timeout=10).json()
		adjclose_list = text['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']
		close_list = text['chart']['result'][0]['indicators']['quote'][0]['close']
		high_list = text['chart']['result'][0]['indicators']['quote'][0]['high']
		low_list = text['chart']['result'][0]['indicators']['quote'][0]['low']
		open_list = text['chart']['result'][0]['indicators']['quote'][0]['open']
		volume_list = text['chart']['result'][0]['indicators']['quote'][0]['volume']
		date_list = text['chart']['result'][0]['timestamp']
		results_orgin = []
		for i in range(len(date_list)):
			date_str = self.stmp2date(date_list[i])
			open1 = str(open_list[i])
			high = str(high_list[i])
			low = str(low_list[i])
			close = str(close_list[i])
			adjclose = str(adjclose_list[i])
			volume = str(volume_list[i])
			tmp = [date_str, open1, high, low, close, adjclose, volume]
			print '|'.join(tmp)
			results_orgin.append([x.encode('gbk') for x in tmp])
		with open('result_orgin.csv', 'a') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerows(results_orgin)
		results_avg = []
		open1_avg = self.get_avg(open_list)
		high_avg = self.get_avg(high_list)
		low_avg = self.get_avg(low_list)
		close_avg = self.get_avg(close_list)
		adjclose_avg = self.get_avg(adjclose_list)
		volume_avg = self.get_avg(volume_list)
		for i in range(len(date_list)):
			date_str = self.stmp2date(date_list[i])
			open1 = self.get_rate(float(open_list[i]), open1_avg)
			high = self.get_rate(float(high_list[i]), high_avg)
			low = self.get_rate(float(low_list[i]), low_avg)
			close = self.get_rate(float(close_list[i]), close_avg)
			adjclose = self.get_rate(float(adjclose_list[i]), adjclose_avg)
			volume = self.get_rate(float(volume_list[i]), volume_avg)
			tmp = [date_str, open1, high, low, close, adjclose, volume]
			print '|'.join(tmp)
			results_avg.append([x.encode('gbk') for x in tmp])
		with open('result_avg.csv', 'a') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerows(results_avg)


if __name__ == "__main__":
	spider = Spider()
	start = '1580774400'
	end = '1591488000'
	spider.get_detail(start, end)
