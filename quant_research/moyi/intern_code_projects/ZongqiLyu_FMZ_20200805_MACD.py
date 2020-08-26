def trade(direction, amount, price):
    if direction == -1 and amount > 0:
        exchange.SetDirection("buy")
        exchange.Buy(price, abs(amount))
    elif direction == -1 and amount < 0:
        exchange.SetDirection("sell")
        exchange.Sell(price, abs(amount))
    if direction == PD_LONG or direction == PD_LONG_YD:
        exchange.SetDirection("closebuy_today" if direction == PD_LONG else "closebuy")
        exchange.Sell(price, abs(amount))
    elif direction == PD_SHORT or direction == PD_SHORT_YD:
        exchange.SetDirection("closesell_today" if direction == PD_SHORT else "closesell")
        exchange.Buy(price, abs(amount))
    
def getTypeAmount():
    while True:
        order = exchange.GetOrders()
        if len(order) != 0:
            for i in range(len(order)):
                exchange.CancelOrder(order[i].Id, order[i])
                Sleep(1000)
        else:
            break
    position = exchange.GetPosition()
    if len(position) == 0:
        return -1, 0
    return position[0].Type, position[0].Amount

def main():
    status_type = -1
    while True:
        if exchange.IO("status"):
            ret = exchange.SetContractType("MA888")
            r = exchange.GetRecords()
            macd = TA.MACD(r)
            if macd[0][-3] is None or macd[1][-3] is None:
                continue
            if macd[0][-3] < macd[1][-3] and macd[0][-2] > macd[1][-2] and (status_type != PD_LONG and status_type != PD_LONG_YD):
                status_type, amount = getTypeAmount()
                if status_type == PD_SHORT or status_type == PD_SHORT_YD:
                    trade(status_type, amount, r[-1]["Close"] + ret["PriceTick"] * 2)
                elif status_type == -1:
                    trade(status_type, Total_amount, r[-1]["Close"] + ret["PriceTick"] * 2)
            elif macd[0][-3] > macd[1][-3] and macd[0][-2] < macd[1][-2] and (status_type != PD_SHORT and status_type != PD_SHORT_YD):
                status_type, amount = getTypeAmount()
                if status_type == PD_LONG or status_type == PD_LONG_YD:
                    trade(status_type, amount, r[-1]["Close"] - ret["PriceTick"] * 2)
                elif status_type == -1:
                    trade(status_type, -Total_amount, r[-1]["Close"] - ret["PriceTick"] * 2)
        else :
            LogStatus(_D(), "未连接")
        Sleep(1000)