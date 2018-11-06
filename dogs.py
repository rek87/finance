#!/usr/bin/env python

#Dogs of the Dow strategy backtesting

import urllib2
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import time
import datetime
import os
import matplotlib.pyplot as plt

def get_data(override=False, filename='dogs.csv'):
    if os.path.exists(filename) and not override:
        return pd.read_csv(filename)

    # Utility function to extract data from stock webpage
    def get_price_by_year(table, yr):
        try:
            return float(table.find(text = 'Jan 01, %4d' % (yr)).find_parent('tr').find_all('td')[1].text)
        except:
            return np.nan

    # Get the symbols from the website
    url_base = "http://www.dogsofthedow.com/"
    pages = []
    #pages = ["dogs2008.htm"]
    #pages = ["dogs%2d.htm" % y for y in range(96, 100)]
    #pages = pages + ["dogs%4d.htm" % y for y in range(2000, 2014)]
    pages = pages + ["dogs%4d.htm" % y for y in range(2008, 2014)]
    pages = pages + ["%4d-dogs-of-the-dow.htm" % y for y in range(2014, 2018)]

    symbols = []
    for p in pages:
        url = url_base + p
        print "url %s" % (url)
        doc = urllib2.urlopen(url)
        html = BeautifulSoup(doc, 'lxml')
        # Look for different headings0
        table = html.find(string=re.compile("NYSE")).find_parent("table")
        lines = table.find_all("tr")[2:]
        symbols.append([l.td.text for l in lines])

    #Loop over the symbols, getting the price at 1/1 and dividend payed in the year
    y = 2008
    df = pd.DataFrame(columns=['Year', 'Symbol', 'Price', 'PriceNext', 'Dividend', 'URL'])
    for s_y in symbols:
        for s in s_y:
            #Scrape data from yahoo finance, not found an API/library to do it
            # URL format
            # https://finance.yahoo.com/quote/MO/history?period1=820450800&period2=851986800&interval=1mo&filter=history&frequency=1mo
            start_ts = int(time.mktime(datetime.date(y,1,1).timetuple()))
            end_ts = int(time.mktime(datetime.date(y+1,1,31).timetuple()))
            url = "https://finance.yahoo.com/quote/%s/history?period1=%d&period2=%d&interval=1mo&filter=history&frequency=1mo" % (s, start_ts, end_ts)

            doc = urllib2.urlopen(url)
            html = BeautifulSoup(doc, 'lxml')

            dividends = 0.0
            i_pr = np.nan
            e_pr = np.nan

            #Get table with data
            table = html.find_all('table', {'data-test': 'historical-prices'})
            if len(table) == 1:
                table = table[0]

                # Get dividend rows
                div_rows = table.find_all(text = 'Dividend')
                for r in div_rows:
                    dividends = dividends + float(r.find_parent('td').strong.text)
                
                # Get prices
                i_pr = get_price_by_year(table, y)
                e_pr = get_price_by_year(table, y+1)

            df = df.append({'Year': y, 'Symbol': s, 'Price': i_pr, 'PriceNext': e_pr, 'Dividend': dividends, 'URL': url}, ignore_index=True)
        y = y + 1

    if filename is not None:
        df.to_csv(filename, index=False)
    return df

def calc_earnings(in_filename='dogs.csv', filename='dogs_full.csv', override=False):
    if os.path.exists(filename) and not override:
        return pd.read_csv(filename)

    in_data = pd.read_csv(in_filename) if os.path.exists(in_filename) else get_data(filename=None)

    # Calculate #stocks and total earnings by year
    tot = pd.DataFrame()
    for y in range(2008, 2018):
        # Current amount: previous total earning or 1000$

        amount = tot[tot['Year'] == (y-1)]['CumulEarn'].iloc[0] / 10 if y > 2008 else 1000
        print "y %d amount: %d" % (y, amount)
        cur_in = in_data.loc[in_data['Year'] == y]
        
        # Built a dataframe with current year data
        cur = pd.DataFrame(cur_in[['Symbol', 'Year']])
        # Calculate number of stocks (integer!)
        cur['Stocks'] = (amount / cur_in['Price']).fillna(0).astype('int64')
        #Calculate initial amount and remaining for the year
        cur['InitAmount'] = (cur['Stocks'] * cur_in['Price']).fillna(0)
        cur['Remaining'] = amount - cur['InitAmount']
        # Calculate earnings from dividends
        cur['DividendEarn'] = cur['Stocks'] * cur_in['Dividend']
        # Calculate Earnings from stock selling at end of year
        cur['StockEarn'] = cur['Stocks'] * cur_in['PriceNext']
        # Calculate total earning for the stock
        cur['TotalEarn'] = (cur['StockEarn'] + cur['DividendEarn']).fillna(0)
        # Calculate total year's earnings
        cur = cur.assign(CumulEarn = (cur['TotalEarn'].sum() + cur['Remaining'].sum()))
        
        print "================================="
        print "===============   %4d  =============" % y
        print cur

        tot = tot.append(cur, ignore_index=True)

    m = pd.merge(in_data, tot, on=['Symbol', 'Year'])
    if filename is not None:
        m.to_csv('dogs_full.csv', index=False)
    return m

def plot(data):
    ax = data[['Symbol','TotalEarn', 'Remaining']].plot(kind='bar', stacked=True, x='Symbol')
    (data['InitAmount']+data['Remaining']).plot(x=data['Symbol'], ax=ax)
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(), (data['Year']%2).values[np.newaxis], cmap='Blues', alpha=.3)
    plt.show()