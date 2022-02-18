import pandas as pd
import numpy as np
import tinvest as ti
import datetime as dt
import time
import asyncio
import trading_functions as tf
import joblib as jl


class predict():
    
    def __init__(self,token,broker_account_id,model,ticker_num=499,except_tickers=[],in_tickers=[],minutes=360,use_sandbox=True, sleep = 0.2501):
        self.figi_vv=None
        self.figi_day_indicators=None
        self.figi_day_candles=None
        self.figi_hour_indicators=None
        self.figi_hour_candles=None
        self.figi_min15_indicators=None
        self.figi_min15_candles=None
        self.figi_min5_indicators=None
        self.figi_min5_candles=None
        self.ticker_num=ticker_num
        self.token=token
        self.use_sandbox=use_sandbox
        self.broker_account_id=broker_account_id
        self.client=ti.SyncClient(token,use_sandbox=use_sandbox)
        stocks=self.client.get_market_stocks().payload.instruments
        self.stocks=np.array([[s.ticker,s.name,str(s.currency),s.figi,s.isin,str(s.type)] for s in stocks])
        self.columns=['Open', 'High', 'Low', 'Close', 'Volume', 
                 'MACD_min5','MACD_slow_min5', 'RSI_min5', 'VWAP!_min5',
                 'Stochastic_min5', 'EMA!_min5', 'CCI_min5', 'ADX_min5',
                 'ADX_delta_min5', 'MFI_min5', 'MACD_min15', 'MACD_slow_min15', 'RSI_min15', 'VWAP!_min15',
                 'Stochastic_min15', 'EMA!_min15', 'CCI_min15', 'ADX_min15',
                 'ADX_delta_min15', 'MFI_min15', 'Open_min15', 'High_min15', 'Low_min15',
                 'Close_min15', 'Volume_min15', 'MACD_h', 'MACD_slow_h', 'RSI_h', 'VWAP!_h',
                 'Stochastic_h', 'EMA!_h', 'CCI_h', 'ADX_h', 'ADX_delta_h', 'MFI_h',
                 'Open_h', 'High_h', 'Low_h', 'Close_h', 'Volume_h', 'MACD_d', 'MACD_slow_d', 'RSI_d',
                 'Stochastic_d', 'EMA!_d', 'CCI_d', 'ADX_d', 'ADX_delta_d',
                 'MFI_d', 'Open_d', 'High_d', 'Low_d', 'Close_d', 'Volume_d']
        self.cancelled_figis=set()
        self.sample_len=len(self.columns)
        self.sample_dict={}
        self.sample_hist={}
        self.model=model
        self.except_tickers=except_tickers
        self.in_tickers=in_tickers
        self.predict_probs=None
        self.predict_log=None      
        self.minutes=minutes
        self.predict_done=False
        self.close_prices={}
        self.active=True
        self.sleep = sleep
        
    def normalize_sample(self,sample):
        sample_shape = sample.shape
        i=0
        if (len(sample_shape)==2) & (sample_shape[0]==1):
            sample=sample[0]
            i=1
        elif len(sample_shape>2):
            print('Too many dimentions!')

        norm_sample = np.zeros(59)
        norm_sample[0],norm_sample[1],norm_sample[2] = sample[0]/sample[3],sample[1]/sample[3],sample[2]/sample[3]
        norm_sample[25],norm_sample[26],norm_sample[27],norm_sample[28] = sample[25]/sample[3],sample[26]/sample[3],\
                                                                          sample[27]/sample[3],sample[28]/sample[3]
        norm_sample[40],norm_sample[41],norm_sample[42],norm_sample[43] = sample[40]/sample[3],sample[41]/sample[3],\
                                                                          sample[42]/sample[3],sample[43]/sample[3]
        norm_sample[54],norm_sample[55],norm_sample[56],norm_sample[57] = sample[54]/sample[3],sample[55]/sample[3],\
                                                                          sample[56]/sample[3],sample[57]/sample[3]
        norm_sample[3]=1

        norm_sample[4]=sample[4]/(sample[58]/225)
        norm_sample[29]=sample[29]/(sample[58]/225*3)
        norm_sample[44]=sample[44]/(sample[58]/225*3*4)
        norm_sample[58]=1

        if i==1:
            return np.array([norm_sample])
        else:
            return norm_sample

    def day_candles(self):
        # Задаем временные границы запроса дневных свеч
        utcnow=dt.datetime.utcnow()
        year=utcnow.year
        month=utcnow.month
        day=utcnow.day
        to_=dt.datetime(year,month,day,tzinfo=dt.timezone.utc)
        from_=dt.datetime(year,month,day)-dt.timedelta(days=50)
        interval=ti.CandleResolution.day
        figi_candles,figi_not_done,figi_no_shame={},[],[]
        # Запрашиваем дневные свечи всех $-ых тикеров
        for figi in self.stocks[:,3][self.stocks[:,2]=='Currency.usd']:
            iter_start=time.time()
            try:
                candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                candles[:,:5]=np.float64(candles[:,:5])
                if len(candles)>=30:
                    figi_candles[figi]=candles
            except:
                figi_not_done.append(figi)
            time.sleep(np.max([iter_start+self.sleep-time.time(),0]))
        # Запрашиваем дневные свечи тикеров, которые не смогли 
        if len(figi_not_done)>0:
            for figi in figi_not_done:
                iter_start=time.time()
                try:
                    candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                    candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                    candles[:,:5]=np.float64(candles[:,:5])
                    if len(candles)>=30:
                        figi_candles[figi]=candles
                except:
                    figi_no_shame.append(figi)
                time.sleep(np.max([iter_start+self.sleep-time.time(),0]))
        # Фильтруем и сортируем тикеры от лучших к худшим
        if len(figi_candles)>=self.ticker_num:
            figi_vv=[]
            for figi in list(figi_candles.keys()):
                avg_close=np.average(figi_candles[figi][:,3].astype(np.float32))
                ticker = self.stocks[self.stocks[:,3]==figi][0,0]
                if (avg_close>=20) & (avg_close<=200) & (ticker in self.in_tickers) & (ticker not in self.except_tickers):
                    avg_volume=np.average(figi_candles[figi][:,4].astype(np.float32))
                    avg_volatility=np.average(np.array((figi_candles[figi][:,1].astype(np.float32)-\
                                                        figi_candles[figi][:,2].astype(np.float32))\
                                                       /figi_candles[figi][:,3].astype(np.float32),
                                                       dtype=np.float32))
                    figi_vv.append((figi,ticker,avg_volume,avg_volatility,avg_close))
            figi_vv=np.array(figi_vv)
            volume_argsort=figi_vv[:,2].astype(np.float32).argsort()
            volume_ranks=np.zeros_like(volume_argsort)
            volume_ranks[volume_argsort]=np.arange(len(volume_argsort))
            volatility_argsort=figi_vv[:,3].astype(np.float32).argsort()
            volatility_ranks=np.zeros_like(volatility_argsort)
            volatility_ranks[volatility_argsort]=np.arange(len(volatility_argsort))
            figi_quality=volume_ranks#+figi_volatility_rank
            figi_vv=np.hstack([figi_vv,figi_quality.reshape(-1,1)])
            self.figi_vv=figi_vv[np.argsort(figi_quality)][::-1]
        else:
            print('Day candles error. Not enough data received from server')
        # Формируем словарь с значениями актуальных дневных индикаторов
        figi_day_indicators={}
        for figi in self.figi_vv[:self.ticker_num,0]:
            self.sample_hist[figi]=np.array([np.hstack([np.array(['Datetime']),np.array(self.columns)])])
            figi_day_indicators[figi]=tf.indicators_table(figi_candles[figi],res='d')[-1:,:]
            self.sample_dict[figi]=np.zeros((1,self.sample_len))
            self.sample_dict[figi][:,45:] = figi_day_indicators[figi]
        self.figi_day_indicators=figi_day_indicators
        self.figi_day_candles=figi_candles
        self.predict_log=np.vstack([np.arange(len(figi_vv[:self.ticker_num])),self.figi_vv[:self.ticker_num,:2].T])
        self.predict_log=np.hstack([np.zeros(3).reshape(-1,1),self.predict_log])
    
    def hour_candles(self):
        # Задаем временные границы запроса часовых свеч
        utcnow=dt.datetime.utcnow()
        year=utcnow.year
        month=utcnow.month
        day=utcnow.day
        hour=utcnow.hour
        to_=dt.datetime(year,month,day,hour,tzinfo=dt.timezone.utc)
        from_=to_-dt.timedelta(days=7)
        interval=ti.CandleResolution.hour
        figi_candles,figi_not_done={},[]
        # Запрашиваем часовые свечи отобранных тикеров
        for figi in self.figi_vv[:self.ticker_num,0]:
            iter_start=time.time()
            try:
                candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                candles[:,:5]=np.float64(candles[:,:5])
                if len(candles)>=30:
                    figi_candles[figi]=candles
            except:
                figi_not_done.append(figi)
            time.sleep(np.max([iter_start+self.sleep-time.time(),0]))
        # Запрашиваем часовые свечи тикеров, которые не смогли 
        if len(figi_not_done)>0:
            for figi in figi_not_done:
                iter_start=time.time()
                try:
                    candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                    candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                    candles[:,:5]=np.float64(candles[:,:5])
                    if len(candles)>=30:
                        figi_candles[figi]=candles
                except:
                    self.cancelled_figis.add(figi)
                time.sleep(np.max([iter_start+self.sleep-time.time(),0]))
        # Формируем словарь с значениями актуальных часовых индикаторов
        figi_hour_indicators={}
        if len(set(figi_candles.keys()).difference(self.cancelled_figis))>=self.ticker_num*0.1:
            for figi in set(figi_candles.keys()).difference(self.cancelled_figis):
                figi_hour_indicators[figi]=tf.indicators_table(figi_candles[figi],res='h')[-1:,:]
                self.sample_dict[figi][:,30:45] = figi_hour_indicators[figi]
            self.figi_hour_indicators=figi_hour_indicators
        else:
            print('Hour candles error. Not enough good ticker data')
        self.figi_hour_candles=figi_candles
    
    def min15_candles(self):
    # Задаем временные границы запроса 15-минутный свеч
        utcnow=dt.datetime.utcnow()
        year=utcnow.year
        month=utcnow.month
        day=utcnow.day
        hour=utcnow.hour
        minute=utcnow.minute
        interval=ti.CandleResolution.min15
        last_from_ = self.figi_day_candles[list(self.figi_day_indicators.keys())[0]][-1,5]
        last_to_ = last_from_ + dt.timedelta(days=1)
        to_=dt.datetime(year,month,day,hour,tzinfo=dt.timezone.utc)
        from_ = np.max([to_-dt.timedelta(days=1),last_to_])
        figi_candles,figi_not_done={},set()
        # Запрашиваем прошлые 15-минутные свечи тикеров
        for figi in set(self.figi_vv[:self.ticker_num,0]).difference(self.cancelled_figis):
            iter_start=time.time()
            try:
                candles=self.client.get_market_candles(figi,last_from_,last_to_,interval).payload.candles
                if len(candles)>=1:
                    candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                    candles[:,:5]=np.float64(candles[:,:5])
                    figi_candles[figi]=candles
                else:
                    self.cancelled_figis.add(figi)
            except:
                self.cancelled_figis.add(figi)
            time.sleep(np.max([iter_start+self.sleep-time.time(),0]))
        # Запрашиваем текущие 15-минутные свечи тикеров
        for figi in set(self.figi_vv[:self.ticker_num,0]).difference(self.cancelled_figis):
            iter_start=time.time()
            try:
                candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                if len(candles)>=1:
                    candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                    candles[:,:5]=np.float64(candles[:,:5])
                    if (figi_candles[figi][-1,-1]==candles[:,5]).any():
                        j=np.argmax(figi_candles[figi][:,5]==candles[0,5])
                        figi_candles[figi]=np.vstack([figi_candles[figi][:j],candles])
                    else:
                        figi_candles[figi]=np.vstack([figi_candles[figi],candles])
                else:
                    self.cancelled_figis.add(figi)
                if len(figi_candles[figi])<30:
                    self.cancelled_figis.add(figi)
            except:
                self.cancelled_figis.add(figi)
            time.sleep(np.max([iter_start+self.sleep-time.time(),0]))
        # Формируем словарь с значениями актуальных 15-минутных индикаторов
        figi_min15_indicators={}
        if len(set(figi_candles.keys()).difference(self.cancelled_figis))>=self.ticker_num*0.1:
            for figi in set(figi_candles.keys()).difference(self.cancelled_figis):
                figi_min15_indicators[figi]=tf.indicators_table(figi_candles[figi],res='min15')[-1:,:]
                self.sample_dict[figi][:,15:30] = figi_min15_indicators[figi]
            self.figi_min15_indicators=figi_min15_indicators
        else:
            print('Min15 candles error. Not enough good ticker data')
        self.figi_min15_candles=figi_candles
        
        
    def min5_candles(self):
        # Пока не будем менять с целью сделать устойчивой к запуску в "ПН" (будем)
        # Задаем временные границы 1-го запроса 5-минутных свеч
        utcnow=dt.datetime.utcnow()
        year=utcnow.year
        month=utcnow.month
        day=utcnow.day
        hour=utcnow.hour
        minute=utcnow.minute
        interval=ti.CandleResolution.min5
        #figi_candles,figi_not_done,figi_no_shame={},[],[]
        #time.sleep((time.time()//300)*300+303-time.time())
        last_from_ = self.figi_day_candles[list(self.figi_day_indicators.keys())[0]][-1,5]
        last_to_ = last_from_ + dt.timedelta(days=1)
        to_=dt.datetime(year,month,day,hour,minute,tzinfo=dt.timezone.utc)
        from_ = np.max([to_-dt.timedelta(days=1),last_to_])
        figi_candles,figi_not_done={},set()
        start=time.time()
        # Запрашиваем прошлые 5-минутные свечи тикеров
        for figi in set(self.figi_vv[:self.ticker_num,0]).difference(self.cancelled_figis):
            iter_start=time.time()
            try:
                candles=self.client.get_market_candles(figi,last_from_,last_to_,interval).payload.candles
                if len(candles)>=1:
                    candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                    candles[:,:5]=np.float64(candles[:,:5])
                    figi_candles[figi]=candles
                else:
                    self.cancelled_figis.add(figi)
            except:
                self.cancelled_figis.add(figi)
            time.sleep(np.max([iter_start+self.sleep-time.time(),0]))
        # Запрашиваем текущие 5-минутные свечи тикеров
        for figi in set(self.figi_vv[:self.ticker_num,0]).difference(self.cancelled_figis):
            iter_start=time.time()
            try:
                candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                if len(candles)>=1:
                    candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                    candles[:,:5]=np.float64(candles[:,:5])
                    if (figi_candles[figi][-1,-1]==candles[:,5]).any():
                        j=np.argmax(figi_candles[figi][:,5]==candles[0,5])
                        figi_candles[figi]=np.vstack([figi_candles[figi][:j],candles])
                    else:
                        figi_candles[figi]=np.vstack([figi_candles[figi],candles])
                else:
                    self.cancelled_figis.add(figi)
                if len(figi_candles[figi])<30:
                    self.cancelled_figis.add(figi)
            except:
                self.cancelled_figis.add(figi)
            time.sleep(np.max([iter_start+self.sleep-time.time(),0]))
        
        # Формируем словарь с значениями актуальных 5-минутных индикаторов
        figi_min5_indicators={}
        if len(set(figi_candles.keys()).difference(self.cancelled_figis))>=self.ticker_num*0.1:
            for figi in set(figi_candles.keys()).difference(self.cancelled_figis):
                figi_min5_indicators[figi]=tf.indicators_table(figi_candles[figi],res='min5')[-1:,:]
                self.sample_dict[figi][:,:15] = np.hstack([figi_candles[figi][-1:,:5],figi_min5_indicators[figi]])
            self.figi_min5_indicators=figi_min5_indicators
        else:
            print('Min5 candles error. Not enough good ticker data')
        self.figi_min5_candles=figi_candles
        #print('Get first min5 candles and indicators time: ', time.time()-start)
        time.sleep(59)
        return self.predict_log,self.figi_min5_candles,self.cancelled_figis,self.sample_hist,\
               self.sample_dict,self.figi_min5_indicators,self.figi_min15_candles,self.figi_min15_indicators, \
               self.figi_hour_candles,self.figi_hour_indicators

        
    async def reg_min5_candles(self):
        # Задаем временные границы регулярного запроса 5-минутных свеч
        utcnow=dt.datetime.utcnow()
        year=utcnow.year
        month=utcnow.month
        day=utcnow.day
        hour=utcnow.hour
        minute=utcnow.minute
        to_=dt.datetime(year,month,day,hour,minute,tzinfo=dt.timezone.utc)
        from_=to_-dt.timedelta(minutes=21)
        interval=ti.CandleResolution.min5
        figi_not_done,figi_no_shame={},[]
        await asyncio.sleep(303-time.time()%300)
        start=time.time()
        #async with ti.AsyncClient(self.token, use_sandbox=self.use_sandbox) as aclient:
        #with self.client as client:
        while time.time()-start<(60*self.minutes):
            pred_sum=0
            utcnow=dt.datetime.utcnow()
            year=utcnow.year
            month=utcnow.month
            day=utcnow.day
            hour=utcnow.hour
            minute=utcnow.minute
            self.predict_done=False
            to_=dt.datetime(year,month,day,hour,minute,tzinfo=dt.timezone.utc)
            from_=to_-dt.timedelta(minutes=21)
            self.predict_log=np.vstack([self.predict_log,np.hstack([to_,np.zeros(self.predict_log.shape[1]-1)])])
            min15_flag=np.floor(time.time()/60%15)==0
            hour_flag=np.floor(time.time()/60%60)==0
            get_indicators_time,get_min15_time,get_hour_time,norm_time=0,0,0,0
            get_candles_start=time.time()
            # Запрашиваем 5-минутную свечу отобранных тикеров
            for figi in set(self.figi_min5_candles.keys()).difference(self.cancelled_figis):
                iter_start=time.time()
                try:
                    candles=client.get_market_candles(figi,from_,to_,interval).payload.candles
                    if len(candles)>=1:
                        candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                        candles[:,:5]=np.float64(candles[:,:5])
                        if ((self.figi_min5_candles[figi][-1,5]==candles[:,5]).any()):
                            j=np.argmax(self.figi_min5_candles[figi][-1,5]==candles[:,5])
                            self.figi_min5_candles[figi]=np.vstack([self.figi_min5_candles[figi][:-1],candles[j:]])
                    else:
                        self.cancelled_figis.add(figi)
                except:
                    self.cancelled_figis.add(figi)
            get_candles_finish=time.time()
            #print('Get candles time: ',get_candles_finish-get_candles_start)
            # Формируем сэмплы
            for figi in set(self.figi_min5_candles.keys()).difference(self.cancelled_figis):
                get_indicators_start=time.time()
                self.sample_hist[figi] = np.vstack([self.sample_hist[figi],np.zeros(self.sample_hist[figi].shape[1])])
                self.sample_hist[figi][-1,0]=to_
                self.figi_min5_indicators[figi]=tf.indicators_table(self.figi_min5_candles[figi][:,-30:],res='min5')[-1:,:]
                self.sample_dict[figi][:,:15]=np.hstack([self.figi_min5_candles[figi][-1:,:5],self.figi_min5_indicators[figi]])
                get_indicators_finish=time.time()
                get_indicators_time+=get_indicators_finish-get_indicators_start
                if min15_flag:
                    get_min15_start=time.time()
                    arr_min15=\
                    self.figi_min5_candles[figi][self.figi_min5_candles[figi][:,5]>=\
                                                 dt.datetime(year,month,day,hour,minute,tzinfo=dt.timezone.utc)-\
                                                 dt.timedelta(minutes=15)
                                                ]
                    if len(arr_min15)>=1:
                        arr_min15=\
                        np.array([arr_min15[0,0],arr_min15[:,1].max(),arr_min15[:,2].min(),arr_min15[-1,3],
                                  arr_min15[:,4].sum(),arr_min15[0,5]])
                        self.figi_min15_candles[figi] = np.vstack([self.figi_min15_candles[figi],arr_min15])
                        self.figi_min15_indicators[figi] = tf.indicators_table(self.figi_min15_candles[figi],res='min15')[-1:,:]
                        self.sample_dict[figi][:,15:30] = self.figi_min15_indicators[figi]
                    else:
                        self.cancelled_figis.add(figi)
                    get_min15_finish=time.time()
                    get_min15_time+=get_min15_finish-get_min15_start
                if hour_flag:
                    get_hour_start=time.time()
                    arr_hour=\
                    self.figi_min5_candles[figi][self.figi_min5_candles[figi][:,5]>=\
                                                 dt.datetime(year,month,day,hour,minute,tzinfo=dt.timezone.utc)-\
                                                 dt.timedelta(hours=1)
                                                ]
                    if len(arr_hour)>=1:
                        arr_hour=\
                        np.array([arr_hour[0,0],arr_hour[:,1].max(),arr_hour[:,2].min(),arr_hour[-1,3],
                                  arr_hour[:,4].sum(),arr_hour[0,5]])
                        self.figi_hour_candles[figi] = np.vstack([self.figi_hour_candles[figi],arr_hour])
                        self.figi_hour_indicators[figi] = tf.indicators_table(self.figi_hour_candles[figi],res='h')[-1:,:]
                        self.sample_dict[figi][:,30:45] = self.figi_hour_indicators[figi]
                    else:
                        self.cancelled_figis.add(figi)
                    get_hour_finish=time.time()
                    get_hour_time+=get_hour_finish-get_hour_start                
                sample=self.sample_dict[figi]
                self.sample_hist[figi][-1,1:]=sample
                self.close_prices[figi]=np.float32(sample[3])
                norm_start=time.time()
                norm_sample=self.normalize_sample(sample)
                norm_finish=time.time()
                norm_time+=norm_finish-norm_start
                #sample = pd.DataFrame(self.sample_dict[figi],columns=self.columns)
                pred_sum+=self.model.predict(norm_sample)[0]
                cur_pred_proba=self.model.predict_proba(norm_sample)[0,1]
                #pred_proba_sum+=cur_pred_proba
                self.predict_log[-1,self.predict_log[1,:]==figi]=cur_pred_proba
                #self.predict_probs.append(cur_pred_proba)
            self.predict_done=True
            #print('Get indicators time: ',get_indicators_time)
            #print('Get min15 indicators time: ',get_min15_time)
            #print('Get hour indicators time: ',get_hour_time)
            #print('Normaliation time: ',norm_time)
            #print('Cancelled figis: ',len(self.cancelled_figis))
            #print(len(set(self.sample_dict.keys()).difference(self.cancelled_figis)),dt.datetime.utcnow())
            print('Number of tickers: %s. Positive predictions: %s. Max probability: %s. Max ticker: %s. Time: %s s.'%(len(set(self.sample_dict.keys()).difference(self.cancelled_figis)),
                      int(pred_sum), 
                      np.round(np.max(np.float64(self.predict_log[-1,1:])),3),
                      self.predict_log[2,np.argmax(self.predict_log[-1,1:])+1],
                      np.round(time.time()%300,1))
                     )
            await asyncio.sleep(303-time.time()%300)

        self.active=False    
            