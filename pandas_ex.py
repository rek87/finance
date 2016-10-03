


grps = data.groupby(data.index) 
[(g[0], g[1].Time[0], g[1].Time[-1], g[1].Open[0], max(g[1].High), min(g[1].Low), g[1].Close[-1]) for g in grps]
oneD = pd.DataFrame.from_records(one_day_data, columns=['Date', 'Start', 'End', 'Open', 'High', 'Low', 'Close'], index='Date')

#Clacola High - Open => Minimum gain
oneD['HO'] = (oneD['High'] - oneD['Open'])*10000

#Calcola Open - Low => Stop Loss


import pandas as pd

data = pd.read_csv("/home/rek/DAT_ASCII_EURGBP_M1_201605.csv", names=['Date', 'Open', 'High', 'Low', 'Close', 'Vol'], index_col=[0], parse_dates=True, sep=';')
a = data.ix[0]
data.index[0]
pd.date_range(start='2016-05-01', freq='s', periods=60)