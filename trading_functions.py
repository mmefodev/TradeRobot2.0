import pandas as pd
import numpy as np
import datetime as dt
import time
import matplotlib.pyplot as plt
import tinvest as ti

# Летнее/зимнее время
class ws():
    
    def __init__(self):
        self.borders=None
    
    def winter_summer(self,input_date,start=dt.datetime.date(dt.datetime(2017,1,1)),finish=dt.datetime.date(dt.datetime(2022,1,1))):
        if type(self.borders)!=np.ndarray:
            date = start
            borders = []

            while date<finish:
                if date.month==4:
                    if (date.day>7) & (date.day<=14) & (date.weekday()==6):
                        borders.append(date)
                        borders.append(date)
                elif date.month==11:
                    if (date.day<=7) & (date.weekday()==6):
                        borders.append(date)
                        borders.append(date)
                date+=dt.timedelta(days=1)

            borders = np.hstack([np.array([start]),np.array(borders),np.array([finish])]).reshape(-1,2)
            w_or_s = np.zeros(borders.shape[0])
            w_or_s[1::2]=1
            self.borders = np.hstack([borders,w_or_s.reshape(-1,1)])


        filter_w_or_s = np.apply_along_axis(lambda x: (input_date>=x[0])&(input_date<x[1]),1,self.borders)
        return np.int8(self.borders[filter_w_or_s][0,2])  


def str_to_datetime(string):
    return dt.datetime.strptime(string,'%Y-%m-%d %H:%M:%S%z')


def EMA(values,period=14,ema_values=None):
    values=np.array(values)
    
    a = (2/(period+1))
    if (type(values)==list) | (type(values)==np.ndarray):
        if (ema_values==None) & (len(values)>1):
            ema_values = [values[0]]
            for i in range(1,len(values)):
                ema_values.append(a*values[i] + (1-a)*ema_values[i-1])
    else:
        if (ema_values==None):
            return 'Incorrect input. Enter EMA value or more close values.'
        elif (len([ema_values])==1) & (len([values])==1):
            ema_values = a*values + (1-a)*ema_values
    
    return np.array(ema_values)


def RSI(data, period=14):
    if type(data)==np.ndarray:
        close_values = data[:,3]
        #arr = np.array(df[['Datetime','Open','High','Low','Close']])
    else:
        close_values = np.array(data['Close'])
    #rsi_values = []
    
    
    close_values_diff = np.hstack([0,close_values[1:] - close_values[:-1]])
        
    upwards = [np.max([x,0]) for x in close_values_diff] 
    downwards = [np.abs(np.min([x,0])) for x in close_values_diff]
    
    EMA_of_U = EMA(upwards,period)
    EMA_of_D = EMA(downwards,period)
    
    rsi_values = 100-100/(1+EMA_of_U/np.where(EMA_of_D==0,0.0001,EMA_of_D))
    
#     for i in range(period,len(arr)):
#         rsi_values.append(100-100/(1+EMA_of_U[i-period]/EMA_of_D[i-period]))
            
#     rsi_values = [None]*period + rsi_values
    
    return np.array(rsi_values)


def detect_min(values,indices=False):
    min_values,min_indices = np.array([None]*len(values)),[]
    
    queue = np.array([0,None])
    
    start=0
    while values[start]==None:
        start+=1
    
    for i in range(start+1,len(values)):
        if values[i-1]>values[i]:
            queue[0],queue[1]=values[i],i
        elif (values[i-1]<values[i])&(queue[1]!=None):
            min_values[queue[1]] = queue[0]
            if indices:
                min_indices.append(np.int32(queue[1]))
    
    if indices:
        return min_values,min_indices
    else:
        return min_values
    
    
def detect_convergention(data,signal,range_=20,buy_indices=True):
    low_values = np.array(data['Low'])
    signal_values = np.array(signal)
    max_low = np.max(low_values)+0.01
    
    low_mins = detect_min(low_values)
    signal_mins = detect_min(signal_values)
    
    
    conv_lows, conv_signals = np.array([None]*len(low_values)), np.array([None]*len(low_values))
    
    low_value1, signal_value1 = None, None
    
    buy_indices_list = []
    
    for i in range(range_,len(low_values)):
        if (low_mins[i]!=None) & (signal_mins[i]!=None):
            low_value2, signal_value2 = low_values[i], signal_values[i]
            double_mins = (low_mins[i-range_:i-1]!=None)&(signal_mins[i-range_:i-1]!=None)
            if double_mins.any():
                #double_mins=[low_mins[i-range_:i-1]==np.min(low_mins[i-range_:i-1][double_mins])]
                #j = i-1-np.argmin(np.where(low_mins[i-range_:i-1]==None,max_low,low_mins[i-range_:i-1])[double_mins][::-1])-1
                j = i-1-np.argmax(double_mins[::-1])-1
                low_value1, signal_value1 = low_values[j],signal_values[j]
            else:
                low_value1, signal_value1 = None, None
        else:
            low_value1, signal_value1, low_value2, signal_value2 = None, None, None, None
        
        if low_value1!=None:
            if (low_value1>low_value2) & (signal_value1<signal_value2):
                conv_lows[j],conv_lows[i],conv_signals[j],conv_signals[i] = \
                low_value1,low_value2,signal_value1,signal_value2
                if buy_indices:
                    buy_indices_list.append(i+1)
    
    if buy_indices:
        return conv_lows, conv_signals, np.array(buy_indices_list)
    else:
        return conv_lows, conv_signals
    
    
def plot_candles(data,start,finish):
    fig, ax1 = plt.subplots(1,1)
    ax_twin = ax1.twiny()
    fig.set_figheight(5)
    fig.set_figwidth(17)
    intervals,count = np.unique(np.vectorize(str_to_datetime)(data.loc[start+1:finish,'Datetime'])-np.vectorize(str_to_datetime)(data.loc[start:finish-1,'Datetime']),return_counts=True)
    interval = intervals[np.argmax(count)]
    datetime_array = np.array([[x,x.year,x.month,x.day,x.hour,x.minute,x.second] for x in np.vectorize(str_to_datetime)(data.loc[start:finish,'Datetime'])])
    ax1.set_title(str(datetime_array[0,1])+'-'+str(datetime_array[0,2])+'-'+str(datetime_array[0,3])+' - '+
                 str(datetime_array[-1,1])+'-'+str(datetime_array[-1,2])+'-'+str(datetime_array[-1,3])
                 )
    
#     ax1.xaxis.set_major_locator(mticker.NullLocator())
#     ax_twin.plot(datetime_array[:,0],[None]*(finish-start+1))
#     ax_twin.xaxis.set_ticks_position('bottom')
    
#     ax_twin.xaxis.set_major_locator(mticker.FixedLocator(datetime_array[:,0]))
#     #ax1.set_xticklabels(list(np.apply_along_axis(lambda x: str(x[4])+':'+str(x[5]),1,datetime_array)))
#     ax_twin.xaxis.set_major_formatter(mticker.FixedFormatter(list(np.apply_along_axis(lambda x: str(x[4])+':'+str(x[5]),1,datetime_array))))
#     #plt.xticks(rotation=45)
    for i in range(start,finish):

        if data.loc[i,'Open']>data.loc[i,'Close']:
            color='r'
        elif data.loc[i,'Open']<data.loc[i,'Close']:
            color='g'
        elif data.loc[i,'Open']==data.loc[i,'Close']:
            color='black'
        
        ax1.plot([datetime_array[i-start,0],datetime_array[i-start,0]],[data.loc[i,'Open'],data.loc[i,'Close']],color,linewidth=3)
        ax1.plot([datetime_array[i-start,0],datetime_array[i-start,0]],[data.loc[i,'Low'],data.loc[i,'High']],color,linewidth=1)
    #plt.show()
    return ax1
            

def check_next(df,indices,depth=10,candle_state='Close'):
    arr = np.array(df[['Datetime','Open','High','Low','Close']])
    
    values = []
    
    if ((len(arr) - indices[-1]) < depth+3):
        indices = indices[:-1]
    
    if candle_state=='Close':
        col = 4
    elif candle_state=='Open':
        col = 1
    elif candle_state=='High':
        col = 2
    elif candle_state=='Low':
        col = 3
    else:
        return 'What a fuck'
    
    for n in range(depth+1):
        indices = list(int(x+n) for x in indices)
        value = np.average(arr[indices,col].astype('float64'))
        values.append(value)
        
    result = list(np.float64(round(values[i]/values[0],3)) for i in range(1,len(values)))
    
    return result


def SMA(values,period=14):
    sma_values = []
    values = np.float64(values)
    for i in range(1,len(values)+1):
        if i<len(values)+1:
            #print(type(np.average(values[np.max([i-period,0]):i])))
            sma_values.append(np.average(values[np.max([i-period,0]):i]))
        elif i==len(values)+1:
            sma_values.append(np.average(values[np.max([i-period,0]):]))
    
    return np.array(sma_values)


def MACD(data,fast_period=12,slow_period=26,signal_period=9):
    if type(data)==np.ndarray:
        close_values=data[:,3]
    else:
        close_values=data['Close']
    #arr = np.array(df[['Datetime','Open','High','Low','Close']])
    
    FastMA = EMA(close_values,fast_period)
    SlowMA = EMA(close_values,slow_period)
    Signal = np.hstack([np.array([0]),FastMA[1:] - SlowMA[1:]])
    SignalMA = np.hstack([np.array([0]),SMA(FastMA[1:]-SlowMA[1:],signal_period)])
    
    return np.array(FastMA,dtype=np.float64),np.array(SlowMA,dtype=np.float64),np.array(Signal,dtype=np.float64),np.array(SignalMA,dtype=np.float64)


def Stochastic(data,K_period=5,K_slow_period=3,D_period=3):
    if type(data)==np.ndarray:
        arr=data[:,1:4]
    else:
        arr=np.array(data[['High','Low','Close']])
        
    #arr = np.array(df[['Datetime','Open','High','Low','Close','Volume']])
    
    FastK_values,SlowK_values,SlowD_values=[],[],[]
    for i in range(K_period-1,len(arr)):
        arr_interval=arr[i-K_period+1:i+1]
        local_max = np.max(arr_interval[:,0])
        local_min = np.min(arr_interval[:,1])
        if (local_max-local_min)!=0:
            FastK = 100*(arr_interval[-1,2]-local_min)/(local_max-local_min)
        else:
            FastK = 100
        FastK_values.append(FastK)
        if len(FastK_values)>=K_slow_period:
            SlowK = np.average(FastK_values[(-1)*K_slow_period:])
            SlowK_values.append(SlowK)
            if len(SlowK_values)>=D_period:
                SlowD = np.average(SlowK_values[(-1)*D_period:])
                SlowD_values.append(SlowD)
    
    FastK_values, SlowK_values, SlowD_values = list(np.array([0]*(K_period-1)))+FastK_values,\
                                               list(np.array([0]*(K_period-1+K_slow_period-1)))+SlowK_values,\
                                               list(np.array([0]*(K_period-1+K_slow_period-1+D_period-1)))+SlowD_values
    
    return np.array(FastK_values,dtype=np.float64), np.array(SlowK_values,dtype=np.float64), np.array(SlowD_values,dtype=np.float64)


def VWAP(data):
    if type(data)==np.ndarray:
        arr=data[:,1:]
    else:
        data['Datetime']=pd.to_datetime(data['Datetime'])
        arr=np.array(data[['High','Low','Close','Volume','Datetime']])
        
    
    #arr = np.array(df[['Datetime','Open','High','Low','Close','Volume']])
    
    vwap_values = []
    new_day = 0
    numerator,denominator = 0,0
    last=dt.datetime(2000,1,1)
    
    for row in arr:
        price = (row[0]+row[1]+row[2])/3
        numerator+=price*row[3]
        denominator+=row[3]
        vwap_values.append(numerator/denominator)
        if (row[4].year!=last.year) | (row[4].day!=last.day):
            numerator,denominator = 0,0
        last=row[4]
    
    return np.array(vwap_values,dtype=np.float64)


def CCI(data,period=14,k=1/0.015):
    if type(data)==np.ndarray:
        arr=data[:,1:4]
    else:
        arr=np.array(data[['High','Low','Close']])
    
    typical_prices = np.float64((arr[:,0]+arr[:,1]+arr[:,2])/3)
    tp_SMA = SMA(typical_prices,period)
    MAD = SMA(np.abs(tp_SMA-typical_prices),period)
    CCI_values = k*(typical_prices-tp_SMA)/np.where(MAD==0,0.0001,MAD)
    return CCI_values


def ADX(data,period=14,k=100,dmi=False):
    if type(data)==np.ndarray:
        arr=data[:,1:4]
    else:
        arr=np.array(data[['High','Low','Close']])

    plus_M = np.hstack([np.zeros(1),arr[1:,0]-arr[:-1,0]])
    minus_M = np.hstack([np.zeros(1),arr[:-1,1]-arr[1:,1]])
    plus_DM = np.where((plus_M>minus_M)&(plus_M>0),plus_M,0)
    minus_DM = np.where((plus_M<minus_M)&(minus_M>0),minus_M,0)
    TR = np.hstack([np.ones(1),np.max([arr[:-1,2],arr[1:,0]],axis=0)-np.min([arr[:-1,2],arr[1:,1]],axis=0)])
    TR = np.where(TR==0,1,TR)
    plus_DI = EMA(plus_DM/TR,period)
    minus_DI = EMA(minus_DM/TR,period)
    DI_summ = np.where((plus_DI+minus_DI)==0,1,(plus_DI+minus_DI))
    ADX_values = k*EMA(np.abs(plus_DI-minus_DI)/DI_summ,period)
    if dmi:
        return ADX_values,plus_DI,minus_DI
    else:
        return ADX_values


def MFI(data,period=14,k=100):
    if type(data)==np.ndarray:
        arr=data[:,1:5]
    else:
        arr=np.array(data[['High','Low','Close','Volume']])

    typical_prices = np.float64((arr[:,0]+arr[:,1]+arr[:,2])/3)
    delta_tp = np.hstack([np.zeros(1),typical_prices[1:]-typical_prices[:-1]])
    PMF = np.where(delta_tp>=0,typical_prices*arr[:,3],0)
    NMF = np.where(delta_tp<0,typical_prices*arr[:,3],0)
    MR = SMA(PMF,period)/np.where(SMA(NMF,period)==0,1,SMA(NMF,period))
    MFI = k-k/(1+MR)
    return MFI


def indicators_table(data,res=''):
    
    FastMA,SlowMA,MACD_values,SignalMA=MACD(data,fast_period=7,slow_period=15,signal_period=5)
    RSI_values=RSI(data)
    VWAP_values=VWAP(data)
    FastK,Stochastic_values,SlowD=Stochastic(data,14,1,3)
    EMA_values=EMA(data[:,3])
    CCI_values=CCI(data,k=1/0.015/100)
    ADX_values=ADX(data,k=1)
    ADX_deltas = np.hstack([np.zeros(1),ADX_values[1:]-ADX_values[:-1]])
    MFI_values=MFI(data,k=1)
    Open_values,High_values,Low_values,Close_values,Volume_values=\
    np.array(data[:,0]),np.array(data[:,1]),np.array(data[:,2]),np.array(data[:,3]),np.array(data[:,4])
    #Range_values = (High_values-Low_values)
    #print(len(MACD_values),len(RSI_values),len(VWAP_values),len(Stochastic_values),len(EMA_values),len(Close_values))
    
    if res=='min5':
        arr = np.vstack([
                      MACD_values,
                      SignalMA,
                      RSI_values/100,
                      VWAP_values/Close_values-1,
                      Stochastic_values/100,
                      EMA_values/Close_values-1,
                      CCI_values,
                      ADX_values,
                      ADX_deltas,
                      MFI_values
                      #Open_values,High_values,Low_values,Close_values,Volume_values
                      #(High_values-Close_values)/np.where(Range_values==0,1,Range_values)
                     ]).T
        
    elif res=='d':
        arr = np.vstack([
                      MACD_values,
                      SignalMA,
                      RSI_values/100,
                      #VWAP_values/Close_values-1,
                      Stochastic_values/100,
                      EMA_values/Close_values-1,
                      CCI_values,
                      ADX_values,
                      ADX_deltas,
                      MFI_values,
                      Open_values,High_values,Low_values,Close_values,Volume_values
                      #(High_values-Close_values)/np.where(Range_values==0,1,Range_values)
                     ]).T
        
    else:
        arr = np.vstack([
                      MACD_values,
                      SignalMA,
                      RSI_values/100,
                      VWAP_values/Close_values-1,
                      Stochastic_values/100,
                      EMA_values/Close_values-1,
                      CCI_values,
                      ADX_values,
                      ADX_deltas,
                      MFI_values,
                      Open_values,High_values,Low_values,Close_values,Volume_values
                      #(High_values-Close_values)/np.where(Range_values==0,1,Range_values)
                     ]).T
        
    return np.float64(arr)


def indicators_table_old(data,res=''):
    data['Open'],data['High'],data['Low'],data['Close'],data['Volume'],data['Datetime']=\
                pd.to_numeric(data['Open']),pd.to_numeric(data['High']),pd.to_numeric(data['Low']),pd.to_numeric(data['Close']),\
                pd.to_numeric(data['Volume']),pd.to_datetime(data['Datetime'])
    FastMA,SlowMA,MACD_values,SignalMA=MACD(data,fast_period=7,slow_period=15,signal_period=5)
    RSI_values=RSI(data)
    VWAP_values=VWAP(data)
    FastK,Stochastic_values,SlowD=Stochastic(data,14,1,3)
    EMA_values=EMA(data['Close'])
    CCI_values=CCI(data,k=1/0.015/100)
    ADX_values=ADX(data,k=1)
    ADX_deltas = np.hstack([np.zeros(1),ADX_values[1:]-ADX_values[:-1]])
    MFI_values=MFI(data,k=1)
    Open_values,High_values,Low_values,Close_values,Volume_values=\
    np.array(data['Open']),np.array(data['High']),np.array(data['Low']),np.array(data['Close']),np.array(data['Volume'])
    #Range_values = (High_values-Low_values)
    #print(len(MACD_values),len(RSI_values),len(VWAP_values),len(Stochastic_values),len(EMA_values),len(Close_values))
    if res=='d':
        datetime_list=pd.to_datetime(data['Datetime'],format='%Y-%m-%d %H:%M:%S')
        datetime_list=np.vectorize(lambda x: x-dt.timedelta(hours=x.hour))(datetime_list)
    else:
        datetime_list=pd.to_datetime(data['Datetime'],format='%Y-%m-%d %H:%M:%S')
    if res=='min5':
        arr = np.vstack([datetime_list,
                      MACD_values,
                      RSI_values/100,
                      VWAP_values/Close_values-1,
                      Stochastic_values/100,
                      EMA_values/Close_values-1,
                      CCI_values,
                      ADX_values,
                      ADX_deltas,
                      MFI_values
                      #Open_values,High_values,Low_values,Close_values,Volume_values
                      #(High_values-Close_values)/np.where(Range_values==0,1,Range_values)
                     ]).T
        df = pd.DataFrame(arr,columns=[res+'_join','MACD_'+res,'RSI_'+res,'VWAP!_'+res,'Stochastic_'+res,'EMA!_'+res,
                                      'CCI_'+res,'ADX_'+res,'ADX_delta_'+res,'MFI_'+res])
        df = df.astype({'MACD_'+res: 'float32',
               'RSI_'+res: 'float32',
               'VWAP!_'+res: 'float32',
               'Stochastic_'+res: 'float32',
               'EMA!_'+res: 'float32',
               'CCI_'+res: 'float32',
               'ADX_'+res: 'float32',
               'ADX_delta_'+res: 'float32',
               'MFI_'+res: 'float32'
              })
    elif res=='d':
        arr = np.vstack([datetime_list,
                      MACD_values,
                      RSI_values/100,
                      #VWAP_values/Close_values-1,
                      Stochastic_values/100,
                      EMA_values/Close_values-1,
                      CCI_values,
                      ADX_values,
                      ADX_deltas,
                      MFI_values,
                      Open_values,High_values,Low_values,Close_values,Volume_values
                      #(High_values-Close_values)/np.where(Range_values==0,1,Range_values)
                     ]).T
        df = pd.DataFrame(arr,columns=[res+'_join','MACD_'+res,'RSI_'+res,'Stochastic_'+res,'EMA!_'+res,
                                      'CCI_'+res,'ADX_'+res,'ADX_delta_'+res,'MFI_'+res,
                                      'Open_'+res,'High_'+res,'Low_'+res,'Close_'+res,'Volume_'+res])
        df = df.astype({'MACD_'+res: 'float32',
               'RSI_'+res: 'float32',
               'Stochastic_'+res: 'float32',
               'EMA!_'+res: 'float32',
               'CCI_'+res: 'float32',
               'ADX_'+res: 'float32',
               'ADX_delta_'+res: 'float32',
               'MFI_'+res: 'float32',
               'Open_'+res: 'float32',
               'High_'+res: 'float32',
               'Low_'+res: 'float32',
               'Close_'+res: 'float32',
               'Volume_'+res: 'int32'
              })
    else:
        arr = np.vstack([datetime_list,
                      MACD_values,
                      RSI_values/100,
                      VWAP_values/Close_values-1,
                      Stochastic_values/100,
                      EMA_values/Close_values-1,
                      CCI_values,
                      ADX_values,
                      ADX_deltas,
                      MFI_values,
                      Open_values,High_values,Low_values,Close_values,Volume_values
                      #(High_values-Close_values)/np.where(Range_values==0,1,Range_values)
                     ]).T
        df = pd.DataFrame(arr,columns=[res+'_join','MACD_'+res,'RSI_'+res,'VWAP!_'+res,'Stochastic_'+res,'EMA!_'+res,
                                      'CCI_'+res,'ADX_'+res,'ADX_delta_'+res,'MFI_'+res,
                                      'Open_'+res,'High_'+res,'Low_'+res,'Close_'+res,'Volume_'+res])
        df = df.astype({'MACD_'+res: 'float32',
               'RSI_'+res: 'float32',
               'VWAP!_'+res: 'float32',
               'Stochastic_'+res: 'float32',
               'EMA!_'+res: 'float32',
               'CCI_'+res: 'float32',
               'ADX_'+res: 'float32',
               'ADX_delta_'+res: 'float32',
               'MFI_'+res: 'float32',
               'Open_'+res: 'float32',
               'High_'+res: 'float32',
               'Low_'+res: 'float32',
               'Close_'+res: 'float32',
               'Volume_'+res: 'int32'
              })
    return df


class candles_request():
    
    def __init__(self,client):
        self.figi_vv=None
        self.figi_day_indicators=None
        self.figi_day_candles=None
        self.figi_hour_indicators=None
        self.figi_hour_candles=None
        self.figi_min15_indicators=None
        self.figi_min15_candles=None
        self.figi_min5_indicators=None
        self.figi_min5_candles=None
        self.client=client
        stocks = client.get_market_stocks().payload.instruments
        self.stocks = np.array([[s.ticker,s.name,str(s.currency),s.figi,s.isin,str(s.type)] for s in stocks])
        self.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
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
        self.sample_len=len(self.columns)
        self.sample_dict = {} #pd.DataFrame(np.zeros((1,len(columns))),columns=columns)
        
    def day_candles(self):
        # Задаем временные границы запроса дневных свеч
        utcnow=dt.datetime.utcnow()
        year=utcnow.year
        month=utcnow.month
        day=utcnow.day
        to_=dt.datetime(year,month,day)
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
            time.sleep(np.max([iter_start+0.1201-time.time(),0]))
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
                time.sleep(np.max([iter_start+0.1201-time.time(),0]))
        # Фильтруем и сортируем тикеры от лучших к худшим
        if len(figi_candles)>=500:
            figi_vv=[]
            for figi in list(figi_candles.keys()):
                avg_close=np.average(figi_candles[figi][:,3].astype(np.float32))
                if avg_close<=500:
                    avg_volume=np.average(figi_candles[figi][:,4].astype(np.float32))
                    avg_volatility=np.average(np.array((figi_candles[figi][:,1].astype(np.float32)-\
                                                        figi_candles[figi][:,2].astype(np.float32))\
                                                       /figi_candles[figi][:,3].astype(np.float32),
                                                       dtype=np.float32))
                    figi_vv.append((figi,avg_volume,avg_volatility,avg_close))
            figi_vv=np.array(figi_vv)
            volume_argsort=figi_vv[:,1].astype(np.float32).argsort()
            volume_ranks=np.zeros_like(volume_argsort)
            volume_ranks[volume_argsort]=np.arange(len(volume_argsort))
            volatility_argsort=figi_vv[:,2].astype(np.float32).argsort()
            volatility_ranks=np.zeros_like(volatility_argsort)
            volatility_ranks[volatility_argsort]=np.arange(len(volatility_argsort))
            figi_quality=volume_ranks#+figi_volatility_rank
            figi_vv=np.hstack([figi_vv,figi_quality.reshape(-1,1)])
            self.figi_vv=figi_vv[np.argsort(figi_quality)][::-1]
        else:
            print('Day candles error. Not enough data received from server')
        # Формируем словарь с значениями актуальных дневных индикаторов
        figi_day_indicators={}
        for figi in self.figi_vv[:500,0]:
            #df=pd.DataFrame(figi_candles[figi],columns=['Open','High','Low','Close','Volume','Datetime'])
            figi_day_indicators[figi]=indicators_table(figi_candles[figi],res='d')[-1:,:]
            self.sample_dict[figi]=np.zeros((1,self.sample_len))#pd.DataFrame(np.zeros((1,len(self.columns))),columns=self.columns)
            # day
            self.sample_dict[figi][:,45:] = figi_day_indicators[figi]
        self.figi_day_indicators=figi_day_indicators
        self.figi_day_candles=figi_candles
    
    def hour_candles(self):
        # Задаем временные границы запроса часовых свеч
        utcnow=dt.datetime.utcnow()
        year=utcnow.year
        month=utcnow.month
        day=utcnow.day
        hour=utcnow.hour
        to_=dt.datetime(year,month,day,hour)
        from_=to_-dt.timedelta(days=7)
        interval=ti.CandleResolution.hour
        figi_candles,figi_not_done,figi_no_shame={},[],[]
        # Запрашиваем часовые свечи отобранных тикеров
        for figi in self.figi_vv[:500,0]:
            iter_start=time.time()
            try:
                candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                candles[:,:5]=np.float64(candles[:,:5])
                #print(len(candles),candles[-1])
                if len(candles)>=30:
                    figi_candles[figi]=candles
            except:
                figi_not_done.append(figi)
            time.sleep(np.max([iter_start+0.1201-time.time(),0]))
        # Запрашиваем часовые свечи тикеров, которые не смогли 
        if len(figi_not_done)>0:
            for figi in figi_not_done:
                iter_start=time.time()
                try:
                    candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                    candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                    candles[:,:5]=np.float64(candles[:,:5])
                    #print(len(candles),candles[-1])
                    if len(candles)>=30:
                        figi_candles[figi]=candles
                except:
                    figi_no_shame.append(figi)
                time.sleep(np.max([iter_start+0.1201-time.time(),0]))
        # Формируем словарь с значениями актуальных часовых индикаторов
        figi_hour_indicators={}
        if len(list(figi_candles.keys()))>=460:
            for figi in list(figi_candles.keys()):
                #df=pd.DataFrame(figi_candles[figi],columns=['Open','High','Low','Close','Volume','Datetime'])
                figi_hour_indicators[figi]=indicators_table(figi_candles[figi],res='h')[-1:,:]
                # hour
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
        to_=dt.datetime(year,month,day,hour)
        from_=to_-dt.timedelta(days=1)
        interval=ti.CandleResolution.min15
        figi_candles,figi_not_done,figi_no_shame={},[],[]
        # Запрашиваем 15-минутные свечи отобранных тикеров
        for figi in self.figi_vv[:500,0]:
            iter_start=time.time()
            try:
                candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                candles[:,:5]=np.float64(candles[:,:5])
                #print(len(candles),candles[-1])
                if len(candles)>=30:
                    figi_candles[figi]=candles
            except:
                figi_not_done.append(figi)
            time.sleep(np.max([iter_start+0.1201-time.time(),0]))
        # Запрашиваем 15-минутные свечи тикеров, которые не смогли 
        if len(figi_not_done)>0:
            for figi in figi_not_done:
                iter_start=time.time()
                try:
                    candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                    candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                    candles[:,:5]=np.float64(candles[:,:5])
                    #print(len(candles),candles[-1])
                    if len(candles)>=30:
                        figi_candles[figi]=candles
                except:
                    figi_no_shame.append(figi)
                time.sleep(np.max([iter_start+0.1201-time.time(),0]))
        # Формируем словарь с значениями актуальных 15-минутных индикаторов
        figi_min15_indicators={}
        if len(list(figi_candles.keys()))>=420:
            for figi in list(figi_candles.keys()):
                #df=pd.DataFrame(figi_candles[figi],columns=['Open','High','Low','Close','Volume','Datetime'])
                figi_min15_indicators[figi]=indicators_table(figi_candles[figi],res='min15')[-1:,:]
                # min15
                self.sample_dict[figi][:,15:30] = figi_min15_indicators[figi]
            self.figi_min15_indicators=figi_min15_indicators
        else:
            print('Min15 candles error. Not enough good ticker data')
        self.figi_min15_candles=figi_candles
        
        
    def min5_candles(self):
        # Задаем временные границы 1-го запроса 5-минутных свеч
        utcnow=dt.datetime.utcnow()
        year=utcnow.year
        month=utcnow.month
        day=utcnow.day
        hour=utcnow.hour
        minute=utcnow.minute
        to_=dt.datetime(year,month,day,hour,minute)
        from_=to_-dt.timedelta(minutes=720)
        interval=ti.CandleResolution.min5
        figi_candles,figi_not_done,figi_no_shame={},[],[]
        time.sleep((time.time()//300)*300+303-time.time())
        start=time.time()
        # Запрашиваем 5-минутные свечи отобранных тикеров
        for figi in list(self.figi_min15_indicators.keys()):
            iter_start=time.time()
            try:
                candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                candles[:,:5]=np.float64(candles[:,:5])
                if len(candles)>=30:
                    figi_candles[figi]=candles
            except:
                figi_not_done.append(figi)
        # Формируем словарь с значениями актуальных 5-минутных индикаторов
        figi_min5_indicators={}
        if len(list(figi_candles.keys()))>=440:
            for figi in list(figi_candles.keys()):
                #df=pd.DataFrame(figi_candles[figi],columns=['Open','High','Low','Close','Volume','Datetime'])
                figi_min5_indicators[figi]=indicators_table(figi_candles[figi],res='min5')[-1:,:]
                self.sample_dict[figi][:,:15] = figi_min5_indicators[figi]
            self.figi_min5_indicators=figi_min5_indicators
        else:
            print('Min5 candles error. Not enough good ticker data')
        self.figi_min5_candles=figi_candles
        print('Get first min5 candles and indicators time: ', time.time()-start)
        
        
    def reg_min5_candles(self):
        # Задаем временные границы регулярного запроса 5-минутных свеч
        utcnow=dt.datetime.utcnow()
        year=utcnow.year
        month=utcnow.month
        day=utcnow.day
        hour=utcnow.hour
        minute=utcnow.minute
        to_=dt.datetime(year,month,day,hour,minute)
        from_=to_-dt.timedelta(minutes=16)
        interval=ti.CandleResolution.min5
        figi_not_done,figi_no_shame=[],[]
        time.sleep((time.time()//300)*300+303-time.time())
        start=time.time()
        while time.time()-start<60*60*6:
            utcnow=dt.datetime.utcnow()
            year=utcnow.year
            month=utcnow.month
            day=utcnow.day
            hour=utcnow.hour
            minute=utcnow.minute
            min15_flag = np.floor(time.time()/60%15)==0
            hour_flag = np.floor(time.time()/60%60)==0
            get_indicators_time,get_min15_time,get_hour_time,norm_time = 0,0,0,0
            get_candles_start=time.time()
            # Запрашиваем 5-минутную свечу отобранных тикеров
            for figi in list(self.figi_min5_candles.keys()):
                iter_start=time.time()
                try:
                    candles=self.client.get_market_candles(figi,from_,to_,interval).payload.candles
                    candles=np.array([[ca.o,ca.h,ca.l,ca.c,ca.v,ca.time] for ca in candles])
                    candles[:,:5]=np.float64(candles[:,:5])
                    if (self.figi_min5_candles[figi][-1,-1]==candles[:,-1]).any():
                        j=np.argmax(self.figi_min5_candles[figi][-1,-1]==candles[:,-1])
                        self.figi_min5_candles[figi]=np.vstack([self.figi_min5_candles[figi][:-1],candles[j:]])
                    else:
                        figi_not_done.append(figi)
                except:
                    figi_not_done.append(figi)
            get_candles_finish=time.time()
            print('Get candles time: ',get_candles_finish-get_candles_start)
            # Формируем словарь с значениями актуальных 5-минутных индикаторов
            for figi in list(self.figi_min5_candles.keys()):
                get_indicators_start=time.time()
                self.figi_min5_indicators[figi]=indicators_table(self.figi_min5_candles[figi][:,-30:],res='min5')[-1:,:]
                get_indicators_finish=time.time()
                get_indicators_time+=get_indicators_finish-get_indicators_start
                
                if min15_flag:
                    get_min15_start=time.time()
                    arr_min15 = \
                    self.figi_min5_candles[figi][self.figi_min5_candles[figi][:,5]>=\
                                                 dt.datetime(year,month,day,hour,minute,tzinfo=dt.timezone.utc)-dt.timedelta(minutes=15)
                                                ]
                    arr_min15 = \
                    np.array([arr_min15[0,0],arr_min15[:,1].max(),arr_min15[:,2].min(),arr_min15[-1,3],arr_min15[:,4].sum(),arr_min15[0,5]])
                    self.figi_min15_candles[figi] = np.vstack([self.figi_min15_candles[figi],arr_min15])
                    self.figi_min15_indicators[figi] = indicators_table(self.figi_min15_candles[figi],res='min15')[-1:,:]
                    self.sample_dict[figi][:,15:30] = self.figi_min15_indicators[figi]
                    get_min15_finish=time.time()
                    get_min15_time+=get_min15_finish-get_min15_start
                    
                if hour_flag:
                    get_hour_start=time.time()
#                     df_hour = \
#                     pd.pivot_table(df[(df['Datetime']>=dt.datetime(year,month,day,hour,tzinfo=dt.timezone.utc)-dt.timedelta(hours=1))&
#                                       (df['Datetime']<dt.datetime(year,month,day,hour,tzinfo=dt.timezone.utc))
#                                      ],
#                                    values=['Open','High','Low','Close','Volume','Datetime'], index=pd.Series(np.zeros(len(df))),
#                                    aggfunc={'Open': 'first',
#                                             'High': np.max,
#                                             'Low': np.min,
#                                             'Close': 'last',
#                                             'Volume': np.sum,
#                                             'Datetime': 'first'
#                                            }
#                                   )
                    arr_hour = \
                    self.figi_min5_candles[figi][self.figi_min5_candles[figi][:,5]>=\
                                                 dt.datetime(year,month,day,hour,minute,tzinfo=dt.timezone.utc)-dt.timedelta(hours=1)
                                                ]
                    arr_hour = \
                    np.array([arr_hour[0,0],arr_hour[:,1].max(),arr_hour[:,2].min(),arr_hour[-1,3],arr_hour[:,4].sum(),arr_hour[0,5]])
                    self.figi_hour_candles[figi] = np.vstack([self.figi_hour_candles[figi],arr_hour])
                    self.figi_hour_indicators[figi] = indicators_table(self.figi_hour_candles[figi],res='h')[-1:,:]
                    self.sample_dict[figi][:,30:45] = self.figi_hour_indicators[figi]
                    get_hour_finish=time.time()
                    get_hour_time+=get_hour_finish-get_hour_start
                
                norm_start=time.time()
                # Нормализуем цены по последнему close value младшей свечи (оставляем все)
                self.sample_dict[figi][-1,0],self.sample_dict[figi][-1,1],self.sample_dict[figi][-1,2]=\
                self.sample_dict[figi][-1,0]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,1]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,2]/self.sample_dict[figi][-1,4]
                
                self.sample_dict[figi][-1,25],self.sample_dict[figi][-1,26],\
                self.sample_dict[figi][-1,27],self.sample_dict[figi][-1,28]=\
                self.sample_dict[figi][-1,25]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,26]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,27]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,28]/self.sample_dict[figi][-1,4]
                
                self.sample_dict[figi][-1,40],self.sample_dict[figi][-1,41],\
                self.sample_dict[figi][-1,42],self.sample_dict[figi][-1,43]=\
                self.sample_dict[figi][-1,40]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,41]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,42]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,43]/self.sample_dict[figi][-1,4]
                
                self.sample_dict[figi][-1,54],self.sample_dict[figi][-1,55],\
                self.sample_dict[figi][-1,56],self.sample_dict[figi][-1,57]=\
                self.sample_dict[figi][-1,54]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,55]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,56]/self.sample_dict[figi][-1,4],\
                self.sample_dict[figi][-1,57]/self.sample_dict[figi][-1,4]
                self.sample_dict[figi][-1,4]=1
                
                # Нормализуем объемы по среднему объему вчера
                self.sample_dict[figi][-1,5]=self.sample_dict[figi][-1,5]/(self.sample_dict[figi][-1,59]/225)
                self.sample_dict[figi][-1,29]=self.sample_dict[figi][-1,29]/(self.sample_dict[figi][-1,59]/225*3)
                self.sample_dict[figi][-1,44]=self.sample_dict[figi][-1,44]/(self.sample_dict[figi][-1,59]/225*3*4)
                self.sample_dict[figi][-1,59]=1
                
                norm_finish=time.time()
                norm_time+=norm_finish-norm_start
            
            print('Get indicators time: ',get_indicators_time)
            print('Get min15 indicators time: ',get_min15_time)
            print('Get hour indicators time: ',get_hour_time)
            print('Normaliation time: ',norm_time)
            print(len(self.sample_dict),dt.datetime.utcnow())
            time.sleep((time.time()//300)*300+303-time.time())
            