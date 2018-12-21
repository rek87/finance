#!/usr/bin/env python3

import requests
import pandas_datareader.data as web
import pandas
import datetime as dt

# csv DB
# stock, buy_date, buy_max_price, take_profit, stop_loss, buy_price, sell_date, sell_profit
# Open the db, looking for stock not yet sold, and check if something has to be updated
def check_stocks(filename='data/db.csv'):
    db = pandas.read_csv(filename)
    db.loc[db['sell_profit'].isnull(), ['buy_price', 'sell_date', 'sell_profit']] = db[db['sell_profit'].isnull()].apply(get_data, axis=1)
    db.to_csv(filename, index=False)

def get_data(row):
    buy_date = dt.datetime.strptime(str(row.buy_date), "%Y%m%d")
    data = web.DataReader(row.stock, 'yahoo', buy_date, dt.date.today())
    buy_price = data.iloc[0]['Open']
    if buy_price > row.buy_max_price:
        return pandas.Series({'buy_price': 0, 'sell_date': dt.date.today(), 'sell_profit': 0})
    low = data[data['Low'] <= row.stop_loss]
    high = data[data['High'] >= row.take_profit]
    date_sl = low.iloc[0].name if len(low) > 0 else None
    date_tp = high.iloc[0].name if len(high) > 0 else None

    if date_sl is not None:
        if date_tp is not None:
            return pandas.Series({'buy_price': buy_price, 'sell_date': date_sl, 'sell_profit': row.stop_loss-buy_price}) if \
                date_sl < date_tp else pandas.Series({'buy_price': buy_price, 'sell_date': date_tp, 'sell_profit': row.take_profit-buy_price})
        else:
            return pandas.Series({'buy_price': buy_price, 'sell_date': date_sl, 'sell_profit': row.stop_loss-buy_price})
    else:
        if date_tp is not None:
            return pandas.Series({'buy_price': buy_price, 'sell_date': date_tp, 'sell_profit': row.take_profit-buy_price})
        else:
            return pandas.Series({'buy_price': None, 'sell_date': None, 'sell_profit': None})