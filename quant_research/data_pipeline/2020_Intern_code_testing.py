# # LucaTian project - 20200812 questions
#
#
# # Example from the link:
#
# # https://doc.shinnytech.com/tqsdk/latest/demo/base.html#t60
#
#
# #!/usr/bin/env python
# #  -*- coding: utf-8 -*-
# __author__ = 'chengzhi'
#
# from tqsdk import TqApi
#
# '''
# 如果当前价格大于10秒K线的MA15则开多仓 (使用 insert_order() 函数)
# 如果小于则平仓
# '''
# api = TqApi()
# # 获得 m2105 10秒K线的引用
# klines = api.get_kline_serial("DCE.m2105", 10)
#
# # 判断开仓条件
# while True:
#     api.wait_update()
#     if api.is_changing(klines):
#         ma = sum(klines.close.iloc[-15:]) / 15
#         print("最新价", klines.close.iloc[-1], "MA", ma)
#         if klines.close.iloc[-1] > ma:
#             print("最新价大于MA: 市价开仓")
#             api.insert_order(symbol="DCE.m2105", direction="BUY", offset="OPEN", volume=5)
#             break
# # 判断平仓条件
# while True:
#     api.wait_update()
#     if api.is_changing(klines):
#         ma = sum(klines.close.iloc[-15:]) / 15
#         print("最新价", klines.close.iloc[-1], "MA", ma)
#         if klines.close.iloc[-1] < ma:
#             print("最新价小于MA: 市价平仓")
#             api.insert_order(symbol="DCE.m2105", direction="SELL", offset="CLOSE", volume=5)
#             break
# # 关闭api,释放相应资源
# api.close()



#t60 - 双均线策略

# from tqsdk import TqApi
# api = TqApi()
# klines = api.get_kline_serial("DCE.m2105",10)
#
#
# while True :
#     api.wait_update()
#     if api.is_changeing(kline):
#         ma = sum(kline.close.iloc[-15:])/15
#         print("latest price", klines.close.iloc[-1],"MA",ma)
#         if kline.close.iloc[-1]>ma:
#             print("latest price grater than MA: buy")
#             api.insert_order(symbol="DEC.m2105", driection ="BUY", offset="OPEN", volume=5)
#             break
#         elif kline.close.iloc[-1]<ma:
#             print("latest price smaller than MA: sell")
#             api.insert_order(symbol="DEC.m2105", driection ="SELL", offset="CLOSE", volume=5)
#             break
#         elif kline.close.iloc[-1] == ma:
#             print("latest price equal to MA")
#             break
# api.close()

# ------------------------------------------------------------
# # LucaTian project updated on 20200818

from datetime import date
from tqsdk import TqApi, TqBacktest, TargetPosTask, TqSim
from tqsdk import BacktestFinished
import matplotlib.pyplot as plt

acc = TqSim()
accountamount = list()
try:
    api = TqApi(acc, backtest=TqBacktest(start_dt=date(2018, 5, 1), end_dt=date(2018, 5, 5)))
    klines = api.get_kline_serial("DCE.m1901", 3600, data_length=15)
    '''target_pos = TargetPosTask(api, "DCE.m1901")'''
    account = api.get_account()

    while True:
        api.wait_update()
        if api.is_changing(klines):
            ma = sum(klines.close.iloc[-15:]) / 15
            print("最新价", klines.close.iloc[-1], "MA", ma)
            if klines.close.iloc[-1] > ma:
                print("最新价大于MA：市价开仓")
                api.insert_order(symbol="DCE.m1901", direction="BUY", offset="OPEN", volume=5)
                accountamount.append(account.balance)



            elif klines.close.iloc[-1] < ma:
                print("最新价小于MA：市价平仓")
                api.insert_order(symbol="DCE.m1901", direction="SELL", offset="CLOSE", volume=5)
                accountamount.append(account.balance)



except BacktestFinished as e:
    print(acc.trade_log)

plt.plot(list(range(1,len(accountamount)+1)), accountamount)
plt.xlabel("number of trade")
plt.ylabel("balance")
plt.show()

api.close()
