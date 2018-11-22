#!/usr/bin/env python3

from bs4 import BeautifulSoup
import pickle
import requests
import pandas_datareader.data as web
import datetime as dt
import ta

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)

    return tickers

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2017, 11, 1)
    end = dt.datetime(2017, 10, 31)
    tribonacci = [7, 13, 24, 44]
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            # Calculate indicators
            for t in tribonacci:
                # Use Open, since I never know whether to use Close or AdjClose
                df['MA%d'%t] = ta.volatility.bollinger_mavg(t['Open'], t)
                df['EMA%d'%t] = ta.trend.ema_indicator(t['Open'], t)
                df['RSI%d'%t] = ta.momentum.rsi(t['Open'], t)
                df['M%d'%t] = t['Open'].diff(t)
            
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))
