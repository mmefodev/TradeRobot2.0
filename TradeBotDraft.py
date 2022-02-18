import asyncio
import numpy as np
import time
import tinvest as ti
import async_predict as p
import joblib as jl
import datetime as dt
import trading_functions as tf
import tracemalloc
tracemalloc.start()


token =  # Your token
broker_account_id =  # Your account id
use_sandbox = True
sleep = 0.2501

cr = ti.CandleResolution.min5
cur_c, todo, _pending, orders, portfolio = {}, set(), set(), np.array([]), np.array([])
lgbmc_min20_pm = jl.load('Models\lgbmc_min20_18_08_21.jl')


NASDAQ_arr = np.genfromtxt('NASDAQ.csv', delimiter=',', dtype=str)
NYSE_arr = np.genfromtxt('NYSE.csv', delimiter=',', dtype=str)
ny_tickers = set(np.hstack([NASDAQ_arr[1:, 0], NYSE_arr[1:]]))
liquid_tickers = set([t[1:] for t in np.genfromtxt('liquid.csv', delimiter=',', dtype=str)[:, 0]])
in_tickers = list(ny_tickers.intersection(liquid_tickers))
streamings = {}
predict_done = False
predict_active, trade_active = True, True
close_prices = {}
trade_log = []
figis = []
figis_trading = []
session_end_time = 0
trade_count = 0
task_count = 0
order_count = 0

client = ti.SyncClient(token, use_sandbox=use_sandbox)
stocks = client.get_market_stocks().payload.instruments
stocks = np.array([[s.ticker, s.name, str(s.currency), s.figi, s.isin,
                    str(s.type), np.float32(s.min_price_increment)] for s in stocks])


def ticker_by_figi(name):
    global stocks
    suffix=''
    if name[-1]=='!':
        suffix='!'
    return str(stocks[:, 0][stocks[:, 3]==name[:12]][0])+suffix


def normalize_sample(sample):
    sample_shape = sample.shape
    i = 0
    if (len(sample_shape) == 2) & (sample_shape[0] == 1):
        sample = sample[0]
        i = 1
    elif len(sample_shape > 2):
        print('Too many dimentions!')

    norm_sample = np.zeros(59)
    norm_sample[0], norm_sample[1], norm_sample[2] = sample[0] / sample[3], sample[1] / sample[3], sample[2] / sample[3]
    norm_sample[25], norm_sample[26], norm_sample[27], norm_sample[28] = sample[25] / sample[3], sample[26] / sample[3], \
                                                                         sample[27] / sample[3], sample[28] / sample[3]
    norm_sample[40], norm_sample[41], norm_sample[42], norm_sample[43] = sample[40] / sample[3], sample[41] / sample[3], \
                                                                         sample[42] / sample[3], sample[43] / sample[3]
    norm_sample[54], norm_sample[55], norm_sample[56], norm_sample[57] = sample[54] / sample[3], sample[55] / sample[3], \
                                                                         sample[56] / sample[3], sample[57] / sample[3]
    norm_sample[3] = 1

    norm_sample[4] = sample[4] / (sample[58] / 225)
    norm_sample[29] = sample[29] / (sample[58] / 225 * 3)
    norm_sample[44] = sample[44] / (sample[58] / 225 * 3 * 4)
    norm_sample[58] = 1

    if i == 1:
        return np.array([norm_sample])
    else:
        return norm_sample


def cancel_old(figi):
    global todo, trade_count, figis_trading, task_count
    trade_count -= 1
    figis_trading.remove(figi)
    for t in todo:
        if t.get_name() == figi+'_'+str(task_count):
            t.cancel()
    # cur_c.pop(figi)
    for t in todo:
        if t.get_name() == figi+'_'+str(task_count)+'!':
            t.cancel()


def cancel_old2(figi):
    global _pending, trade_count, figis_trading, task_count
    trade_count -= 1
    figis_trading.remove(figi)
    print(_pending)
    for t in _pending:
        if (t.get_name()[:12] == figi) & (t.get_name()[-1] != '!'):
            print(t)
            t.cancel()
    print(_pending)
    cur_c.pop(figi)
    for t in _pending:
        if (t.get_name()[:12] == figi) & (t.get_name()[-1] == '!'):
            print(t)
            t.cancel()


async def stream_stop_old(figi):
    global _pending, trade_count, figis_trading, streamings
    for t in _pending:
        t_name = t.get_name()
        if (t_name[:12] == figi) & (t_name[-1] != '!'):
            streamings[figi].stop()
    cur_c.pop(figi)
    streamings.pop(figi)
    trade_count -= 1
    figis_trading.remove(figi)


async def stream_stop(figi):
    global streamings, cur_c, trade_count, figis_trading
    streaming_stopped = False
    while not streaming_stopped:
        try:
            await streamings[figi].stop()
            streaming_stopped = True
        except:
            await asyncio.sleep(1-time.time()%1)
    cur_c.pop(figi)
    streamings.pop(figi)
    trade_count -= 1
    figis_trading.remove(figi)


async def quick_sell(figi, aclient, broker_account_id):
    global orders, portfolio
    while figi in orders[:,1]:
        order_ids = orders[:, 0][orders[:, 1] == figi]
        for order_id in order_ids:
            try:
                print('Quick sell. About to cancel order')
                await aclient.post_orders_cancel(order_id, broker_account_id)
                print(ticker_by_figi(figi)+' quick sell. Order cancelled')
            except:
                print(ticker_by_figi(figi)+' quick sell. Order NOT cancelled. Trying again...')
                pass
        await asyncio.sleep(1.1 - time.time() % 1)
    while figi in portfolio[:, 1]:
        try:
            lots=int(portfolio[:, 4][portfolio[:, 1] == figi][0])
            body=ti.MarketOrderRequest(
                lots=lots,
                operation='Sell'
            )
            print('Quick sell. About to post market sell order')
            await aclient.post_orders_market_order(figi, body, broker_account_id)
            print(ticker_by_figi(figi) + ' quick sell. Sell order placed')
            break
        except:
            print(ticker_by_figi(figi) + ' quick sell. Sell order NOT placed. Trying again...')
            pass
        await asyncio.sleep(1.1 - time.time() % 1)
    while (figi in portfolio[:, 1]) | (figi in orders[:, 1]):
        await asyncio.sleep(1.1 - time.time() % 1)


async def stream(token, figi, candle_resolution):
    global cur_c, streamings
    cur_c[figi] = 0
    async with ti.Streaming(token) as streaming:
        streamings[figi] = streaming
        await streaming.candle.subscribe(figi, candle_resolution)
        #await streaming.instrument_info.subscribe(figi)
        async for event in streaming:
            cur_c[figi] = np.float32(event.payload.c)


async def trade_control(figi, token, broker_account_id, tp=0.004, sl=-0.002, start_bound=0.00025, order_period=11,
                        execute_period=35, minute_interval=20, order_size=200, use_sandbox=use_sandbox):
    global cur_c, orders, portfolio, todo, close_prices, trade_count, streamings, order_count
    await asyncio.sleep(0.1)
    buy_order_posted, buy_order_executed, sell_order_posted = False, False, False
    last_sell_time = minute_interval*60+time.time()//300*300
    start_price = np.float32(close_prices[figi])
    ticker = ticker_by_figi(figi)
    start_time = time.time()
    async with ti.AsyncClient(token, use_sandbox=use_sandbox) as aclient:
        await quick_sell(figi, aclient, broker_account_id)
        while not buy_order_posted:
            # buy order conditions
            cur_price = cur_c[figi]
            if (cur_price > (1-start_bound)*start_price) & (cur_price < (1+start_bound)*start_price) & \
               (figi not in orders[:, 1]) & (figi not in portfolio[:, 1]) & (time.time()-start_time < order_period) & \
               (figi in list(cur_c.keys())):
                # order
                try:
                    lots = 1  # int(order_size//start_price)
                    body = ti.LimitOrderRequest(
                        lots=lots,
                        operation='Buy',
                        price=np.float32(np.round(cur_price + 0.01, 2))
                    )
                    await aclient.post_orders_limit_order(figi, body, broker_account_id)
                    buy_order_posted = True
                    order_count += 1
                    print(ticker + ' buy order posted')
                    await asyncio.sleep(1.1 - time.time() % 1)
                except :
                    print(ticker + ' buy order NOT posted. Trying again...')
                    await asyncio.sleep(1.1 - time.time() % 1)
            elif time.time() - start_time >= order_period:
                # cancel bid try, cancel streaming, cancel trade_control
                await quick_sell(figi, aclient, broker_account_id)
                print(ticker + ' trade cancelled')
                await stream_stop(figi)
                break
            await asyncio.sleep(1.1 - time.time() % 1)
        # wait for and check order execution
        start_time = time.time()
        while (buy_order_executed == False) & (buy_order_posted == True):
            if time.time()-start_time < execute_period:
                executed_lots = 0
                for p in portfolio[:, 4][portfolio[:, 1] == figi]:
                    executed_lots = int(p)

                if executed_lots < lots:
                    await asyncio.sleep(1.1 - time.time() % 1)
                elif executed_lots == lots:
                    buy_order_executed = True
                    print(ticker + ' buy order fully executed')
                    buy_price = cur_price
                    await asyncio.sleep(1.1 - time.time() % 1)
            else:
                while figi in orders[:, 1]:
                    order_ids = orders[:, 0][orders[:, 1] == figi]
                    for order_id in order_ids:
                        try:
                            await aclient.post_orders_cancel(order_id, broker_account_id)
                            print(ticker + ' execution time is over. Order cancelled')
                        except:
                            print(ticker + ' execution time is over. Problem with order cancel. Trying again...')
                            pass
                    await asyncio.sleep(1.1 - time.time() % 1)
                executed_lots = 0
                for p in portfolio[:, 4][portfolio[:, 1] == figi]:
                    executed_lots = int(p)

                if executed_lots == 0:
                    await quick_sell(figi,aclient,broker_account_id)
                    print(ticker + ' execution time is over. Buy order NOT executed')
                    await stream_stop(figi)
                    break
                elif executed_lots>0:
                    buy_order_executed=True
                    buy_price = cur_price
                    print(ticker + ' execution time is over. Buy order executed')
                    await asyncio.sleep(1.1-time.time()%1)
        # print(figi+' Buy order executed')
        # waiting for conditions to post sell order, control execution
        while (sell_order_posted==False) & (buy_order_executed==True) & (buy_order_posted==True):
            cur_price = cur_c[figi]
            if (cur_price>=(1+tp)*start_price) | (cur_price<=(1+sl)*start_price) | (time.time()>last_sell_time):
                try:
                    body=ti.LimitOrderRequest(
                        lots=executed_lots,
                        operation='Sell',
                        price=np.float32(np.round(cur_price-0.01,2))
                    )
                    await aclient.post_orders_limit_order(figi, body, broker_account_id)
                    print(ticker + ' sell order posted')
                    sell_order_posted = True
                except:
                    print(ticker + ' sell order NOT posted. Trying again...')
                    pass
            await asyncio.sleep(1.1-time.time()%1)
        # waiting for sell order to execute
        if (sell_order_posted==True) & (buy_order_executed==True) & (buy_order_posted==True):
            limit_sell_flag = True
            while (figi in portfolio[:, 1]) | (figi in orders[:, 1]):
                if time.time() > last_sell_time:
                    await quick_sell(figi,aclient,broker_account_id)
                    limit_sell_flag = False
                await asyncio.sleep(1.1-time.time()%1)
            if limit_sell_flag:
                print(ticker + ' limit sell order executed')
                sell_price = cur_price
            else:
                print(ticker + ' market sell order executed')
                sell_price = cur_c[figi]
            trade_log.append([dt.datetime.utcnow(), ticker, figi, start_price, buy_price, sell_price, executed_lots])
            print('%s trade done. Start price: %s. Buy price: %s. Sell price: %s.'%
                (ticker, start_price, buy_price, sell_price))
            await stream_stop(figi)


async def trade(token, broker_account_id, cr, channels=6, max_orders=145):
    global predict_done, session_end_time, trade_active, todo, figis, trade_count, figis_trading, task_count, \
        _pending, order_count
    trade_active, cur_pred_iter, task_add_done = True, 0, False
    while ((time.time() < session_end_time) | (trade_count > 0)) | (order_count >= max_orders):
        if cur_pred_iter != time.time()//300:
            predict_done, task_add_done = False, False
            print('Orders posted in session: ' + str(order_count))
        if predict_done:
            if not task_add_done:
                for figi in figis:
                    # if (figi not in [p.get_name() for p in _pending]) & (len(_pending)<=5*2):
                    if (trade_count < channels) & (figi not in figis_trading): # len(_pending) <= 5*2:
                        task_count += 1
                        task = asyncio.create_task(stream(token, figi, cr),
                                                   name=figi+'_'+str(task_count)
                                                  )
                        todo.add(task)
                        task = asyncio.create_task(trade_control(figi, token, broker_account_id),
                                                   name=figi+'_'+str(task_count)+'!'
                                                  )
                        todo.add(task)
                        trade_count += 1
                        figis_trading.append(figi)
                        done, _pending = await asyncio.wait(todo, timeout=0.5)
                        todo.difference_update(done)
                        cur_tickers = (ticker_by_figi(t.get_name()) for t in todo)
                        print(f'{len(todo)}: ' + ','.join(sorted(cur_tickers)))
                task_add_done = True
        cur_pred_iter = time.time()//300
        await asyncio.sleep(1.1-time.time()%1)
    jl.dump(trade_log, dt.datetime.now().strftime(format='tl_%d_%m_%y.jl'))
    trade_active = False


async def trade_old(token, broker_account_id, cr):
    global cur_c,todo,predict_active,trade_active,predict_done,figis,trade_log
    done, _pending = [],[]
    add_to_todo_done = False
    # await asyncio.sleep(200)
    while (predict_active) | (len(_pending)>0):
        # if time.time() % 10 // 1 == 0:
        #     print(predict_done,add_to_todo_done)
        # check predict_log, check "in process" list
        if predict_done:
            # jl.dump(trade_log,dt.datetime.now().strftime(format='tl_%d_%m_%y.jl'))
            if not add_to_todo_done:
                # jl.dump(predict_log,'test_predict_log.jl')
                # pos_preds = predict_log[[1,2,-1],1:][:,predict_log[-1,1:]>=bound]
                # figis = pos_preds[0,np.argsort(pos_preds[-1])][::-1]
                for figi in figis:
                    if (figi not in [p.get_name() for p in _pending])&(len(_pending)<=5*2):
                        task = asyncio.create_task(stream(token,figi,cr),
                                                   name=figi
                                                  )
                        todo.add(task)
                        task = asyncio.create_task(trade_control(figi,token,broker_account_id),
                                                   name=figi+'!'
                                                  )
                        todo.add(task)
                        figis.remove(figis[0])
                        done, _pending = await asyncio.wait(todo,timeout=0.5)
                        todo.difference_update(done)
                        cur_tickers=(ticker_by_figi(t.get_name()) for t in todo)
                        print(f'{len(todo)}: '+','.join(sorted(cur_tickers)))
                add_to_todo_done = True
                await asyncio.sleep(303-time.time()%300)
                continue
            await asyncio.sleep(1-time.time()%1)
            continue
        else:
            add_to_todo_done = False
            await asyncio.sleep(1-time.time()%1)
            continue
        # print('trade working...')
        # await asyncio.sleep(1-time.time()%1)
    jl.dump(trade_log,dt.datetime.now().strftime(format='tl_%d_%m_%y.jl'))
    trade_active=False


async def predict(token, use_sandbox, cr, model, sample_hist, predict_log, figi_min5_candles, figi_min5_indicators,
                  figi_min15_candles, figi_min15_indicators, figi_hour_candles, figi_hour_indicators,
                  sample_dict, cancelled_figis, bound):
    # Задаем временные границы регулярного запроса 5-минутных свеч
    global predict_done, figis, close_prices, session_end_time
    utcnow = dt.datetime.utcnow()
    year = utcnow.year
    month = utcnow.month
    day = utcnow.day
    hour = utcnow.hour
    minute = utcnow.minute
    to_ = dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)
    from_ = to_ - dt.timedelta(minutes=21)
    interval = cr
    await asyncio.sleep(303 - time.time() % 300)
    start = time.time()
    client = ti.SyncClient(token, use_sandbox=use_sandbox)
    #async with ti.AsyncClient(token, use_sandbox=use_sandbox) as aclient:
    # with self.client as client:
    while time.time() < session_end_time:
        pred_sum = 0
        utcnow = dt.datetime.utcnow()
        year = utcnow.year
        month = utcnow.month
        day = utcnow.day
        hour = utcnow.hour
        minute = utcnow.minute
        #predict_done = False
        to_ = dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc)
        from_ = to_ - dt.timedelta(minutes=21)
        predict_log = np.vstack([predict_log, np.hstack([to_, np.zeros(predict_log.shape[1] - 1)])])
        min15_flag = np.floor(time.time() / 60 % 15) == 0
        hour_flag = np.floor(time.time() / 60 % 60) == 0
        # get_indicators_time, get_min15_time, get_hour_time, norm_time = 0, 0, 0, 0
        # get_candles_start = time.time()
        # Запрашиваем 5-минутную свечу отобранных тикеров
        for figi in set(figi_min5_candles.keys()).difference(cancelled_figis):
            # iter_start = time.time()
            try:
                candles = client.get_market_candles(figi, from_, to_, interval).payload.candles
                if len(candles) >= 1:
                    candles = np.array([[ca.o, ca.h, ca.l, ca.c, ca.v, ca.time] for ca in candles])
                    candles[:, :5] = np.float64(candles[:, :5])
                    if ((figi_min5_candles[figi][-1, 5] == candles[:, 5]).any()):
                        j = np.argmax(figi_min5_candles[figi][-1, 5] == candles[:, 5])
                        figi_min5_candles[figi] = np.vstack([figi_min5_candles[figi][:-1], candles[j:]])
                    else:
                        figi_min5_candles[figi] = np.vstack([figi_min5_candles[figi], candles])
                else:
                    pass
                    #cancelled_figis.add(figi)
            except:
                cancelled_figis.add(figi)
        # get_candles_finish = time.time()
        # print('Get candles time: ',get_candles_finish-get_candles_start)
        # Формируем сэмплы
        for figi in set(figi_min5_candles.keys()).difference(cancelled_figis):
            # get_indicators_start = time.time()
            sample_hist[figi] = np.vstack([sample_hist[figi], np.zeros(sample_hist[figi].shape[1])])
            sample_hist[figi][-1, 0] = to_
            figi_min5_indicators[figi] = tf.indicators_table(figi_min5_candles[figi][:, -30:], res='min5')[-1:, :]
            sample_dict[figi][:, :15] = np.hstack([figi_min5_candles[figi][-1:, :5], figi_min5_indicators[figi]])
            # get_indicators_finish = time.time()
            # get_indicators_time += get_indicators_finish - get_indicators_start
            if min15_flag:
                # get_min15_start = time.time()
                arr_min15 = \
                    figi_min5_candles[figi][figi_min5_candles[figi][:, 5] >= \
                                                 dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc) - \
                                                 dt.timedelta(minutes=15)
                                                 ]
                if len(arr_min15) >= 1:
                    arr_min15 = \
                        np.array([arr_min15[0, 0], arr_min15[:, 1].max(), arr_min15[:, 2].min(), arr_min15[-1, 3],
                                  arr_min15[:, 4].sum(), arr_min15[0, 5]])
                    figi_min15_candles[figi] = np.vstack([figi_min15_candles[figi], arr_min15])
                    figi_min15_indicators[figi] = tf.indicators_table(figi_min15_candles[figi], res='min15')[-1:, :]
                    sample_dict[figi][:, 15:30] = figi_min15_indicators[figi]
                else:
                    pass
                    # cancelled_figis.add(figi)
                # get_min15_finish = time.time()
                # get_min15_time += get_min15_finish - get_min15_start
            if hour_flag:
                # get_hour_start = time.time()
                arr_hour = \
                    figi_min5_candles[figi][figi_min5_candles[figi][:, 5] >= \
                                                 dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc) - \
                                                 dt.timedelta(hours=1)
                                                 ]
                if len(arr_hour) >= 1:
                    arr_hour = \
                        np.array([arr_hour[0, 0], arr_hour[:, 1].max(), arr_hour[:, 2].min(), arr_hour[-1, 3],
                                  arr_hour[:, 4].sum(), arr_hour[0, 5]])
                    figi_hour_candles[figi] = np.vstack([figi_hour_candles[figi], arr_hour])
                    figi_hour_indicators[figi] = tf.indicators_table(figi_hour_candles[figi], res='h')[-1:, :]
                    sample_dict[figi][:, 30:45] = figi_hour_indicators[figi]
                else:
                    pass
                    # cancelled_figis.add(figi)
                # get_hour_finish = time.time()
                # get_hour_time += get_hour_finish - get_hour_start
            sample = sample_dict[figi]
            sample_hist[figi][-1, 1:] = sample
            close_prices[figi] = np.float32(sample[-1,3])
            # norm_start = time.time()
            norm_sample = normalize_sample(sample)
            # norm_finish = time.time()
            # norm_time += norm_finish - norm_start
            # sample = pd.DataFrame(self.sample_dict[figi],columns=self.columns)
            # pred_sum += model.predict(norm_sample)[0]
            cur_pred_proba = model.predict_proba(norm_sample)[0, 1]
            # pred_proba_sum+=cur_pred_proba
            predict_log[-1, predict_log[1, :] == figi] = cur_pred_proba
            # self.predict_probs.append(cur_pred_proba)
        pos_preds = predict_log[[1, 2, -1], 1:][:, predict_log[-1, 1:] >= bound]
        figis = list(pos_preds[0, np.argsort(pos_preds[-1])][::-1])
        tickers=[ticker_by_figi(figi) for figi in figis]
        if len(tickers)>5:
            tickers=tickers[:5]
            tickers.append('...')
        elif len(tickers)==0:
            tickers=['No tickers']
        else:
            tickers=tickers[:5]
        predict_done = True
        print('Number of tickers: %s. Positive predictions: %s (%s). Time: %s s.' %
              (
               len(set(sample_dict.keys()).difference(cancelled_figis)),
               pos_preds.shape[1],
               # np.round(np.max(np.float64(predict_log[-1, 1:])), 3),
               # predict_log[2, np.argmax(predict_log[-1, 1:]) + 1],
               ','.join(tickers),
               np.round(time.time() % 300, 1))
              )
        await asyncio.sleep(303 - time.time() % 300)

    jl.dump(predict_log,dt.datetime.now().strftime(format='pl_%d_%m_%y.jl'))
    jl.dump(sample_hist, dt.datetime.now().strftime(format='sh_%d_%m_%y.jl'))


async def get_orders_and_portfolio(token, broker_account_id, use_sandbox=use_sandbox):
    global orders,portfolio,trade_active
    #await asyncio.sleep(200)
    async with ti.AsyncClient(token, use_sandbox=use_sandbox) as aclient:
        while trade_active:
            # if time.time()%10//1==0:
            #     print('get_orders_and_portfolio working...')
            try:
                get_orders = await aclient.get_orders(broker_account_id)
                orders = np.array([[order.order_id,order.figi,order.operation,order.requested_lots,order.executed_lots,
                                    order.price,ticker_by_figi(order.figi)]
                                  for order in get_orders.payload])
                if len(orders) == 0:
                    orders = np.array([[0, 'figi.', 'no', 0, 0, 0, 'ticker.']])
            except:
                pass
            try:
                get_portfolio = await aclient.get_portfolio(broker_account_id)
                portfolio = np.array([[position.instrument_type,position.figi,position.ticker,position.balance,
                                       position.lots]
                                  for position in get_portfolio.payload.positions])
                if len(portfolio) == 0:
                    portfolio = np.array([['type.','figi.','ticker.',0,0]])
            except:
                pass
            #print('get_orders_and_portfolio working...')
            await asyncio.sleep(1-time.time()%1)


async def gather_processes(token, broker_account_id, use_sandbox, candle_resolution, predict_bound, model,
                           sample_hist, predict_log, figi_min5_candles, figi_min5_indicators, figi_min15_candles,
                           figi_min15_indicators, figi_hour_candles, figi_hour_indicators, sample_dict,
                           cancelled_figis, minutes, channels, max_orders):
    global session_end_time
    session_end_time = time.time()+minutes*60
    processes = [asyncio.create_task(predict(token,use_sandbox,candle_resolution,model,sample_hist,predict_log,
                                             figi_min5_candles,figi_min5_indicators,figi_min15_candles,
                                             figi_min15_indicators,figi_hour_candles,figi_hour_indicators,
                                             sample_dict,cancelled_figis,predict_bound)),
                 asyncio.create_task(get_orders_and_portfolio(token,broker_account_id,use_sandbox)),
                 asyncio.create_task(trade(token,broker_account_id,candle_resolution,channels,max_orders))
                ]
    await asyncio.gather(*processes)


# main
loop = asyncio.get_event_loop()
pr = p.predict(token, broker_account_id, lgbmc_min20_pm, ticker_num=160, except_tickers=['SPCE', 'TSM'],
               in_tickers=in_tickers, minutes=20, use_sandbox=use_sandbox, sleep=sleep)
pr.day_candles()
pr.hour_candles()
pr.min15_candles()
predict_log, figi_min5_candles, cancelled_figis, sample_hist, \
sample_dict, figi_min5_indicators, figi_min15_candles, figi_min15_indicators, \
figi_hour_candles, figi_hour_indicators = pr.min5_candles()
for figi in list(sample_hist.keys()):
    cur_c[figi] = 0

loop.run_until_complete(gather_processes(token, broker_account_id, use_sandbox, cr, 0.5,
                           lgbmc_min20_pm, sample_hist, predict_log, figi_min5_candles, figi_min5_indicators,
                           figi_min15_candles, figi_min15_indicators, figi_hour_candles,
                           figi_hour_indicators, sample_dict, cancelled_figis, 250, 6, 110))
loop.close()