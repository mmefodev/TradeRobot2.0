import pandas as pd
import numpy as np
import tinvest as ti
import datetime as dt
import time
import random
import joblib as jl

SANDBOX_TOKEN =  # Your sandbox token
client = ti.SyncClient(SANDBOX_TOKEN, use_sandbox=True)
broker_account_id = client.get_accounts().payload.accounts[0].broker_account_id

# NASDAQ_arr = np.genfromtxt('NASDAQ.csv',delimiter=',',dtype=str)
# NYSE_arr = np.genfromtxt('NYSE.csv',delimiter=',',dtype=str)
# ny_tickers = set(np.hstack([NASDAQ_arr[1:,0],NYSE_arr[1:]]))
# liquid_tickers = set([t[1:] for t in np.genfromtxt('liquid.csv',delimiter=',',dtype=str)[:,0]])
# in_tickers = list(ny_tickers.intersection(liquid_tickers))
# liquid_tickers_choice = np.random.choice(in_tickers,50,False)
# jl.dump(liquid_tickers_choice,'liquid_tickers_choice.jl')
# liquid_tickers_choice = jl.load('liquid_tickers_choice.jl')
liquid_tickers_choice = ['DSKY', 'ENRU', 'TGKA', 'PLZL', 'OGKB', 'MOEX', 'SBER', 'GAZP', 'VTBR', 'LKOH', 'ROSN', 'GMKN',
                         'SNGS', 'SBERP', 'FEES', 'HYDR', 'CHMF', 'TRNFP', 'NVTK', 'MGNT', 'TATN', 'NLMK', 'SNGSP',
                         'MTSS', 'RSTI', 'RTKM', 'AFLT', 'IRAO', 'SIBN', 'MAGN', 'MTLR', 'RTKMP', 'AFKS', 'RASP',
                         'BANEP', 'UPRO', 'BANE', 'TATNP', 'PHOR', 'ALRS', 'PIKK', 'LSRG', 'MVID', 'MTLRP', 'RUAL',
                         'CBOM']

# timedelta list: days
timedelta_list_days = []
datetime = dt.datetime(2020,1,1,0)
timedelta_list_days.append(datetime)
while datetime < dt.datetime(2021, 8, 15, 0):
    datetime = datetime + dt.timedelta(days=1)
    timedelta_list_days.append(datetime)

# timedelta list: years
timedelta_list_years = []
datetime = dt.datetime(2020,1,1,0)
timedelta_list_years.append(datetime)
while datetime < dt.datetime(2021, 8, 15, 0):
    if datetime.year%4==0:
        days_number=366
    else:
        days_number=365
    datetime = datetime + dt.timedelta(days=days_number)
    timedelta_list_years.append(datetime)

# Large caps. Hours
interval = ti.CandleResolution.hour
tickers = liquid_tickers_choice
figis = [client.get_market_search_by_ticker(ticker).payload.instruments[0].figi for ticker in tickers]
tickers_figis = list(zip(tickers, figis))
for ticker, figi in tickers_figis:
       candles_array_list = []
       for i in range(1, len(timedelta_list_days)):
              from_ = timedelta_list_days[i - 1]
              to = timedelta_list_days[i]
              candles = client.get_market_candles(figi, from_, to, interval).payload.candles
              candles_array = np.array(list(map(lambda x: [x.time, np.float32(x.o), np.float32(x.h), np.float32(x.l), \
                                                           np.float32(x.c), np.int32(x.v), ticker, figi], candles)))
              if candles_array.all() != np.array([]).all():
                     candles_array_list.append(candles_array)
              time.sleep(0.1201)
       df = pd.DataFrame(np.vstack(candles_array_list), columns= \
              ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'FIGI'])
       df.to_csv(
              'D:\\Pycharm\\PycharmProjects\\TradingRobot\\RuData\\Candles hour\\Candles ' + ticker + ' hour.csv',
              index=False)

# Large caps. Min15
interval = ti.CandleResolution.min15
tickers = liquid_tickers_choice
figis = [client.get_market_search_by_ticker(ticker).payload.instruments[0].figi for ticker in tickers]
tickers_figis = list(zip(tickers,figis))
for ticker,figi in tickers_figis:
    candles_array_list = []
    for i in range(1,len(timedelta_list_days)):
        from_ = timedelta_list_days[i-1]
        to = timedelta_list_days[i]
        candles = client.get_market_candles(figi,from_,to,interval).payload.candles
        candles_array = np.array(list(map(lambda x: [x.time, np.float32(x.o), np.float32(x.h), np.float32(x.l),\
                                         np.float32(x.c), np.int32(x.v), ticker, figi], candles)))
        if candles_array.all() != np.array([]).all():
            candles_array_list.append(candles_array)
        time.sleep(0.1201)
    df = pd.DataFrame(np.vstack(candles_array_list), columns = \
                      ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'FIGI'])
    df.to_csv('D:\\Pycharm\\PycharmProjects\\TradingRobot\\RuData\\Candles min15\\Candles '+ticker+' min15.csv',
              index=False)

# Large caps. Min5
interval = ti.CandleResolution.min5
tickers = liquid_tickers_choice

figis = [client.get_market_search_by_ticker(ticker).payload.instruments[0].figi for ticker in tickers]
tickers_figis = list(zip(tickers,figis))
for ticker,figi in tickers_figis:
    candles_array_list = []
    for i in range(1,len(timedelta_list_days)):
        from_ = timedelta_list_days[i-1]
        to = timedelta_list_days[i]
        candles = client.get_market_candles(figi,from_,to,interval).payload.candles
        candles_array = np.array(list(map(lambda x: [x.time, np.float32(x.o), np.float32(x.h), np.float32(x.l),\
                                         np.float32(x.c), np.int32(x.v), ticker, figi], candles)))
        if candles_array.all() != np.array([]).all():
            candles_array_list.append(candles_array)
        time.sleep(0.1201)
    df = pd.DataFrame(np.vstack(candles_array_list), columns = \
                      ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'FIGI'])
    df.to_csv('D:\\Pycharm\\PycharmProjects\\TradingRobot\\RuData\\Candles min5\\Candles '+ticker+' min5.csv',
              index=False)

# Large caps. Days
interval = ti.CandleResolution.day
tickers = liquid_tickers_choice
figis = [client.get_market_search_by_ticker(ticker).payload.instruments[0].figi for ticker in tickers]
tickers_figis = list(zip(tickers,figis))
for ticker,figi in tickers_figis:
    candles_array_list = []
    for i in range(1,len(timedelta_list_years)):
        from_ = timedelta_list_years[i-1]
        to = timedelta_list_years[i]
        candles = client.get_market_candles(figi,from_,to,interval).payload.candles
        candles_array = np.array(list(map(lambda x: [x.time, np.float32(x.o), np.float32(x.h), np.float32(x.l),\
                                         np.float32(x.c), np.int32(x.v), ticker, figi], candles)))
        if candles_array.all() != np.array([]).all():
            candles_array_list.append(candles_array)
        time.sleep(0.1201)
    df = pd.DataFrame(np.vstack(candles_array_list), columns = \
                      ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'FIGI'])
    df.to_csv('D:\\Pycharm\\PycharmProjects\\TradingRobot\\RuData\\Candles day\\Candles '+ticker+' day.csv',
              index=False)