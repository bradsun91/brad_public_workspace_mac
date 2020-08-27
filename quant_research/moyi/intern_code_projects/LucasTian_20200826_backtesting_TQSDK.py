from datetime import date
import datetime
from tqsdk import TqApi, TqBacktest, TqSim
from tqsdk import BacktestFinished
import matplotlib.pyplot as plt

futures_contracts_name=["CFFEX.T2012","CFFEX.TF2012","SHFE.ag2012","SHFE.au2012","SHFE.ni2010","SHFE.cu2010","SHFE.al2009","SHFE.rb2101","SHFE.zn2010","SHFE.pb2010","SHFE.sn2011","DCE.i2101","SHFE.hc2010","SHFE.ss2010","DCE.jm2101","DCE.j2101","SHFE.sp2012","DCE.m2101","DCE.p2101","DCE.c2101","DCE.eb2101","DCE.eg2101","DCE.p2101","DCE.pp2101","DCE.a2101","DCE.b2010","DCE.y2101","SHFE.fu2101","DCE.pg2011","SHFE.bu2012","DCE.cs2101","DCE.l2101"]
futures_contracts= dict()
api=TqApi()
for i in futures_contracts_name:
    if api._data.quotes[i].margin<20000:
        futures_contracts.update({i: api._data.quotes[i].margin})
    else:
        print(i+" margin is grater than 20000")
futures_contracts_sorted= sorted(futures_contracts.items(),key=lambda x:x[1])
print(futures_contracts_sorted)
futures_contracts_name_sorted=sorted(futures_contracts,key=futures_contracts.__getitem__)
selected_contracts=[futures_contracts_name_sorted[0],futures_contracts_name_sorted[1],futures_contracts_name_sorted[2],futures_contracts_name_sorted[3],futures_contracts_name_sorted[4]]
print(selected_contracts)


for i in selected_contracts:
    acc = TqSim()
    accountamount = list()
    accounttime = list()

    try:
        api = TqApi(TqSim(20000), backtest=TqBacktest(start_dt=date(2020, 3, 1), end_dt=date(2020, 3, 5)))
        klines = api.get_kline_serial(i, 3600, data_length=15)
        account = api.get_account()
        print(i)

        while True:
            api.wait_update()
            if api.is_changing(klines):
                ma = sum(klines.close.iloc[-15:]) / 15
                print("最新价", klines.close.iloc[-1], "MA", ma)
                if klines.close.iloc[-1] > ma:
                    print("最新价大于MA：市价开仓")
                    api.insert_order(symbol=i, direction="BUY", offset="OPEN", volume=1)
                    accountamount.append(account.balance)
                    accounttime.append(datetime.datetime.fromtimestamp(klines.iloc[-1]["datetime"]/1e9))

                elif klines.close.iloc[-1] < ma:
                    print("最新价小于MA：市价平仓")
                    api.insert_order(symbol=i, direction="SELL", offset="CLOSE", volume=1)
                    accountamount.append(account.balance)
                    accounttime.append(datetime.datetime.fromtimestamp(klines.iloc[-1]["datetime"]/1e9))

    except BacktestFinished as e:
        print(acc.trade_log)

    str= input("choose the plot type: according to date or number of trade (please type 'd' or 'n')")

    if str == "d":
        plt.plot(accounttime, accountamount)
        plt.xlabel(i + "   date")
        plt.ylabel(i + "   balance")
        plt.show()
    elif str == "n":
        plt.plot(list(range(1, len(accountamount) + 1)), accountamount)
        plt.xlabel(i + "   number of trade")
        plt.ylabel(i + "   balance")
        plt.show()
    else:
        print("invalid input. please try again.")


api.close()