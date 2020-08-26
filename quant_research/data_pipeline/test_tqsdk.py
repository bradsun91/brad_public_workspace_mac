# 引入TqSdk模块
import csv
# from tqsdk import TqApi
# import pandas as pd, numpy as np

# 创建api实例，设置web_gui=True生成图形化界面
# api = TqApi(web_gui=True)
# # 订阅 cu2002 合约的10秒线
# klines = api.get_kline_serial("SHFE.cu2002", 10)
# # while True:
# #     # 通过wait_update刷新数据
# #     api.wait_update()
#
# df_klines = pd.DataFrame(klines)
#
# ch_db_path = "/Users/miaoyuesun/Code_Workspace/brad_public_workspace_mac/data/CH_database/"
#
# df_klines.to_csv(ch_db_path+"test_tq_klines.csv", index = False)

######################################################################################
# 要获得你的账户资金情况, 可以请求一个资金账户引用对象:
# account = api.get_account()
# print(account)


######################################################################################
# Automated traidng using SMA as an example:

# from tqsdk import TqApi
# from tqsdk.ta import MA
# from datetime import date
# from tqsdk import TqApi, TqSim, TqBacktest
# from tqsdk import BacktestFinished
# import time
#
# def open_positions(klines):
#     MA20 = MA(klines, 5)
#     MA40 = MA(klines, 10)
#     if (MA20.iloc[-1].ma > MA40.iloc[-1].ma) and (MA20.iloc[-2].ma < MA40.iloc[-1].ma):
#         return 1
#     return 0
#
# def close_positions(klins):
#     MA20 = MA(klines, 5)
#     MA40 = MA(klines, 10)
#     if (MA20.iloc[-1].ma < MA40.iloc[-1].ma) and (MA20.iloc[-2].ma > MA40.iloc[-1].ma):
#         return 1
#     return 0
#
# acc = TqSim()
#
# try:
#     api = TqApi(acc, backtest=TqBacktest(start_dt=date(2019, 11, 7), end_dt=date(2019, 12, 31)))
#     # get klines
#     klines = api.get_kline_serial("SHFE.rb2001",3600) # 3600: requesting hourly data
#     while 1:
#         # print(klines)
#         api.wait_update()
#         klines=api.get_kline_serial("SHFE.rb2001", 3600)
#         # print(api.get_position("SHFE.rb2001"))
#         # decide if to open positions
#         if open_positions(klines):
#             # check target positions
#             cur_long_pos = api.get_position("SHFE.rb2001")
#             # check if there're any pending orders
#             cur_orders = api.get_order()
#             if cur_long_pos['pos_long_his'] == 0 and cur_orders =={}:
#                 order = api.insert_order(symbol='SHFE.rb2001',direction="BUY",offset="OPEN",volume=5)
#                 print("Opened the positions!")
#                 # print(klines[''])
#
#         # decide if to close positions
#         if close_positions(klines):
#             cur_long_pos = api.get_position("SHFE.rb2001")
#             if cur_long_pos['pos_long_his']!=0:
#                 order = api.insert_order(symbol='SHFE.rb2001', direction="SELL", offset="CLOSE", volume=5)
#                 print("Closed the positions!")
#
# except BacktestFinished as e:
#     # 回测结束时会执行这里的代码
#     print(acc.trade_log)
#
# api.close()

######################################################################################

# E.g. https://doc.shinnytech.com/tqsdk/latest/demo/base.html#tutorial-backtest

# from datetime import date
# from tqsdk import TqApi, TqBacktest, TargetPosTask, TqSim
# from tqsdk import BacktestFinished
#
# '''
# 如果当前价格大于5分钟K线的MA15则开多仓
# 如果小于则平仓
# 回测从 2018-05-01 到 2018-10-01
# '''
# acc = TqSim()
#
# try:
#     # 在创建 api 实例时传入 TqBacktest 就会进入回测模式
#     api = TqApi(acc, backtest=TqBacktest(start_dt=date(2018, 5, 1), end_dt=date(2018, 5, 5)))
#     # 获得 m1901 1HR K线的引用
#     klines = api.get_kline_serial("DCE.m1901", 3600, data_length=15)
#     # 创建 m1901 的目标持仓 task，该 task 负责调整 m1901 的仓位到指定的目标仓位
#     target_pos = TargetPosTask(api, "DCE.m1901")
#
#     while True:
#         api.wait_update()
#         if api.is_changing(klines):
#             ma = sum(klines.close.iloc[-15:]) / 15
#             print("最新价", klines.close.iloc[-1], "MA", ma)
#             if klines.close.iloc[-1] > ma:
#                 print("最新价大于MA: 目标多头5手")
#                 # 设置目标持仓为多头5手
#                 target_pos.set_target_volume(5)
#             elif klines.close.iloc[-1] < ma:
#                 print("最新价小于MA: 目标空仓")
#                 # 设置目标持仓为空仓
#                 target_pos.set_target_volume(0)
#
# except BacktestFinished as e:
#     # 回测结束时会执行这里的代码
#     print(acc.trade_log)
#
#
# api.close()


######################################################################################

#回测情况下的图形化界面

from datetime import date
from tqsdk import TqApi, TqBacktest, TargetPosTask, TqSim
from tqsdk import BacktestFinished

acc = TqSim()

try:
    # 在创建 api 实例时传入 TqBacktest 就会进入回测模式
    api = TqApi(acc, backtest=TqBacktest(start_dt=date(2018, 5, 1), end_dt=date(2018, 5, 5)))
    # 获得 m1901 1HR K线的引用
    klines = api.get_kline_serial("DCE.m1901", 3600, data_length=15)
    # 创建 m1901 的目标持仓 task，该 task 负责调整 m1901 的仓位到指定的目标仓位
    target_pos = TargetPosTask(api, "DCE.m1901")

    while True:
        api.wait_update()
        if api.is_changing(klines):
            ma = sum(klines.close.iloc[-15:]) / 15
            print("最新价", klines.close.iloc[-1], "MA", ma)
            if klines.close.iloc[-1] > ma:
                print("最新价大于MA: 目标多头5手")
                # 设置目标持仓为多头5手
                target_pos.set_target_volume(5)
            elif klines.close.iloc[-1] < ma:
                print("最新价小于MA: 目标空仓")
                # 设置目标持仓为空仓
                target_pos.set_target_volume(0)

except BacktestFinished as e:
    # 回测结束时会执行这里的代码
    print(acc.trade_log)


api.close()




