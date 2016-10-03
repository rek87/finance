


grps = data.groupby(data.index) 
[(g[0], g[1].Time[0], g[1].Time[-1], g[1].Open[0], max(g[1].High), min(g[1].Low), g[1].Close[-1]) for g in grps]
oneD = pd.DataFrame.from_records(one_day_data, columns=['Date', 'Start', 'End', 'Open', 'High', 'Low', 'Close'], index='Date')

#Clacola High - Open => Minimum gain
oneD['HO'] = (oneD['High'] - oneD['Open'])*10000

#Calcola Open - Low => Stop Loss


import pandas as pd
import numpy as np

#Generate second data from M1 candle
def generate_data(row):
	# Create an array of 60 random number, ranging  [Low,High)
	gdf=np.random.random(60)*(row.High-row.Low)+row.Low
	# Adjust open, close and max values
	gdf[0]=row.Open
	gdf[-1]=row.Close
	gdf[gdf.argmax()]=row.High
	# Return a pandas dataframe with generated data
	return pd.DataFrame(gdf, index=pd.date_range(start=row.XDate, freq='s', periods=60))

#Read data from file
data = pd.read_csv("data/DAT_ASCII_EURGBP_M1_201605.csv", names=['Date', 'Open', 'High', 'Low', 'Close', 'Vol'], index_col=[0], parse_dates=True, sep=';')
data['XDate']=data.index #Replicate the index as a value to use apply
#Generate fulll data
gen_data=[]
for i in range(len(data)):
	if i == 0:
		gen_data = generate_data(data.ix[i])
	else:
		gen_data=gen_data.append(generate_data(data.ix[i]))

#gen_data.to_csv('data/DAT_ASCII_EURGBP_M1_201605_generated.csv')
