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
