#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:40:50 2020

@author: lutian
"""

from tqsdk import TqApi
api = TqApi()
klines = api.get_kline_serial("DCE.m2105",10)


while True :
    api.wait_update()
    if api.is_changeing(kline):
        ma = sum(kline.close.iloc[-15:])/15
        print("latest price", klines.close.iloc[-1],"MA",ma)
        if kline.close.iloc[-1]>ma:
            print("latest price grater than MA: buy")
            api.insert_order(symbol="DEC.m2105", driection ="BUY", offset="OPEN", volume=5)
            break
        elif kline.close.iloc[-1]<ma:
            print("latest price smaller than MA: sell")
            api.insert_order(symbol="DEC.m2105", driection ="SELL", offset="CLOSE", volume=5)
            break
        elif kline.close.iloc[-1] == ma:
            print("latest price equal to MA")
            break
api.close()

    