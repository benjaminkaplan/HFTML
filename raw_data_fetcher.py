# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:52:43 2023

@author: benja
"""
import pandas as pd
import os
from alpaca.data import StockHistoricalDataClient, StockBarsRequest
from alpaca.data.enums import Adjustment
from alpaca.data.timeframe import TimeFrame

tickers = ["SPY"]
#os.chdir("C:\Binyamin\Thesis\HFT_ML")
df = pd.read_csv("alpaca_keys.csv")
API_KEY =df.loc[0,'api_key'] 
SECRET_KEY = df.loc[0,'secret_key']

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

adj = Adjustment("all")
request_params = StockBarsRequest(
                        symbol_or_symbols=tickers,
                        timeframe=TimeFrame.Minute,
                        start="2017-01-01 00:00:00",
                        end="2018-12-31 11:59:59",
                        adjustment=adj
                 ) 

df = client.get_stock_bars(request_params).df
for i in df.index.levels[0]:
    idx = pd.IndexSlice
    name = i +"_minute_2017_2018.csv"
    df.loc[idx[i,:],:].droplevel(level=0).to_csv(name)


"""
#startup
import os
import pandas as pd
import  matplotlib.pyplot as plt
os.chdir("C:\Binyamin\Thesis\HFT_ML")
df = pd.read_csv("AAPL_minute.csv").set_index("timestamp")
df.index = pd.to_datetime(df.index)
df.index = df.index.tz_convert("America/New_York")
"""



