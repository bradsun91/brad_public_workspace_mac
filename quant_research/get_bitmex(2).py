# made by Ajay_Luo
import pandas as pd
import requests
import json
import time


def get_list():
    print('输入格式为y-m-d')
    start = str(input('请输入起始时间：'))
    end = str(input('请输入结束时间：'))
    #name = str(input('请输入名字：'))
    T = pd.date_range(start=start, end=end)
    time_list = [str(list(T)[i]) for i in range(len(T))]
    return time_list  # ,  name

def get_time_period():
    print('时间间隔的格式为: 1m, 5m, 1h, 或者1d')
    timeperiod = str(input('请输入需要爬取的时间间隔: '))
    return timeperiod

def get_json(url):
    try:
        time.sleep(2)
        r = requests.get(url, headers={'user_agent' : 'IE=edge'})
        r.raise_for_status()
        r.encoding='utf-8'
        return r.text
    except:
        print("{}失败".format(url))
        exit()



def get_info(time_list, name, time_period):
    time_period = timeperiod
    for i in range(len(time_list)):
        if time_period == '1h':
            url = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=1h&count=24&partial=false&symbol' \
            '=.'+str(name)+'&reverse=false&startTime='+time_list[i]
        elif time_period == '5m':
            url = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=5m&count=288&partial=false&symbol' \
            '=.'+str(name)+'&reverse=false&startTime='+time_list[i]
        elif time_period == '1d':
            url = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=1d&count=1&partial=false&symbol' \
            '=.'+str(name)+'&reverse=false&startTime='+time_list[i]
          
            
        if time_period != '1m':
            j = get_json(url)
            data_list = json.loads(j)
            data = pd.DataFrame(data_list)
            
        elif time_period == '1m':
            url01 = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=1m&count=720&partial=false&symbol' \
            '=.'+str(name)+'&reverse=false&startTime='+time_list[i]
            
            url02 = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=1m&count=720&partial=false&symbol' \
            '=.'+str(name)+'&reverse=false&startTime='+time_list[i].split(' ')[0] + ' 12:00:00'
            
            j01 = get_json(url01)
            j02 = get_json(url02)
            data_list01 = json.loads(j01)
            data_list02 = json.loads(j02)
            
            data01 = pd.DataFrame(data_list01)
            data02 = pd.DataFrame(data_list02)
            
            data = pd.concat([data01, data02], axis = 0)
        
        if i == 0:
            data.to_csv('/Users/hurenjie/Desktop/同梁志能/Code/比特币代码/抓取的比特币数据/'+str(name)+' D.csv', header=True)
            print(time_list[i])
        else:
            data.to_csv('/Users/hurenjie/Desktop/同梁志能/Code/比特币代码/抓取的比特币数据/'+str(name)+' D.csv', mode='a', header=False)
            print(time_list[i])


name_list = ['BADAXBT', 'BBCHXBT', 'BEOSXBT', 'BETHXBT', 'BLTCXBT', 'BXBT', 'BXRPXBT', 'TRXXBT']

time_list = get_list()

timeperiod = get_time_period()
for name in name_list:
    get_info(time_list, name, timeperiod)
# get_info(time_list, 'XBTU19')


# 1 m, 5 m, 1 h, 1 d
    
























