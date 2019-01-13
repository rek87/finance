#!/usr/bin/env python3

import os
from bs4 import BeautifulSoup
import pickle
import requests
import pandas_datareader.data as web
import pandas
import datetime as dt
import talib as ta
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.decomposition import PCA

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
    end = dt.datetime(2018, 10, 31)
    for ticker in tickers:
        print('Getting {}'.format(ticker))
        # just in case your connection breaks, we'd like to save our progress!

        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
            except:
                print ('Excpetion when retrieving %s' % ticker)
                continue

            df['Stock'] = ticker

            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

def prepare_data():
    all_df = pandas.DataFrame()
    # Loop over the stock_dfs folder, looking for CSVs
    for f in os.listdir('stock_dfs'):
        # Load the data dropping missing values
        df = pandas.read_csv('stock_dfs/%s' % f).dropna()

        periods = [2, 3, 5, 7, 9, 11, 13, 17, 19, 24, 31, 44]
        # Calculate indicators
        for p in periods:
            # I never know whether to use Close or AdjClose...
            #df['I_N_MA%d'%p] = ta.SMA(df['Close'], timeperiod=p)
            df['I_N_MA%d_scaled'%p] =  ta.SMA(df['Close'], timeperiod=p) / df['Close']
            #df['I_N_EMA%d'%p] = ta.EMA(df['Close'], timeperiod=p)
            df['I_N_EMA%d_scaled'%p] = ta.EMA(df['Close'], timeperiod=p) / df['Close']
            df['I_RSI%d'%p] = ta.RSI(df['Close'], timeperiod=p) / 100
            df['I_N_NATR%d'%p] = ta.NATR(df['High'], df['Low'], df['Close'], timeperiod=p)
            #df['M%d'%p] = df['Open'].diff(p)
            df['I_N_M%d_scaled'%p] = df['Open'].diff(p) / df['Close']

        # Add the outcome of a possible transaction: Long, Short, Nothing
        # Checking if in the following 4 weeks the stock can earn the 5%
        # 1st: calculate max and min prices in next 4 weeks
        df['4WMax'] = df.rolling(22).Close.max().shift(-20)
        df['4Wmin'] = df.rolling(22).Close.min().shift(-20)
        # 2nd: calculate market position
        df['Long'] = 0 + ((df.Close * 1.05) < df['4WMax']) # Trick to have numbers
        df['Short'] = 0 + ((df.Close * .95) > df['4Wmin'])
        df.loc[df['Long'] == df['Short'], ['Long', 'Short']] = 0 # Ensure there are not both set
        df['Stay'] = 1 - df['Long'] - df['Short']
        df = df.dropna()
        # 3rd: scale data
        if not df.empty:
            scaler = preprocessing.MinMaxScaler()
            # Select indicator columns
            cols = [ c for c in df.columns if c.find('_N_') > 0]
            try:
                df[cols] = scaler.fit_transform(df[cols])
            except:
                print ('Exception when scalig %s ' % f)
                print df.head()
                continue
            all_df = all_df.append(df.dropna())

    all_df.to_csv('common.csv', index=False)
    return all_df

def create_nn(input_dim):
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_pca_model(filename=None):
    # Get input data
    data = prepare_data() if filename is None else pandas.read_csv(filename).dropna()

    # Shuffle before PCA and normalization
    data = data.sample(frac=1).reset_index(drop=True)

    # Apply PCA with automatic components detection
    pca = PCA(n_components=.999,svd_solver='full')
    reduced_indicators = pca.fit_transform(data[[c for c in data.columns if c.find('I_') >= 0]])
    # Re-scale data in the interval [0:1]
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(reduced_indicators)

    # Get Ys
    Y = data[['Long','Short','Stay']]

    # Create and train model
    model = create_nn(X.shape[1])
    model.fit(X, Y, epochs=10000, batch_size=100, verbose=2)
    return model, pca

def train_model(filename=None):
    # Get input data
    data = prepare_data() if filename is None else pandas.read_csv(filename).dropna()

    # Shuffle before PCA and normalization
    data = data.sample(frac=1).reset_index(drop=True)

    # Get Ys
    Y = data[['Long','Short','Stay']]
    X = data[[c for c in data.columns if c.find('I_') >= 0]]

    # Create and train model
    model = create_nn(X.shape[1])
    model.fit(X, Y, epochs=10000, batch_size=100, verbose=2)
    return model

if __name__ == '__main__':
    get_data_from_yahoo()

# Print correlation
# import seaborn as sns
# import matplotlib.pyplot as plt
# X=data[[ c for c in data.columns if c.find('_N_') > 0]]
# sns.heatmap(red_X.corr())
# plt.show
