import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import padasip as pa

def predictor(s, delay=2, order=3, mu=.98):
    y=np.zeros(len(s)+delay)
    e=np.zeros(len(s))
    filt=pa.filters.FilterNLMS(order, mu=mu)
    for k in range(order+delay-1, len(s)):
        # Calculate y[k-delta] - Useless, is recalculated in adapt method
        p = filt.predict(s[k-delay-order+1:k-delay+1])
        e[k]=s[k]-p
        # Adapt coeff using current x[k] and y[k-delta]
        filt.adapt(s[k], s[k-delay-order+1:k-delay+1])
        y[k+delay] = filt.predict(s[k-order+1:k+1])
    return y, e

def predictor2(s, delay=2, order=3, mu=.98):
    y=np.zeros(len(s)+delay)
    e=np.zeros(len(s))
    filt=pa.filters.FilterNLMS(order, mu=mu, w="zeros")
    x = pa.input_from_history(s, order)
    print len(x)
    for k in range(delay, len(x)):
        p = filt.predict(x[k-delay])
        filt.adapt(s[k+order-1], x[k-delay])
        y[k] = p
    return y

def predictor3(s, delay=2, order=3, mu=.98):
    xd = np.pad(s, (delay, 0), mode='constant')
    y=np.zeros(len(xd))
    e=np.zeros(len(xd))
    filt=pa.filters.FilterNLMS(order, mu=mu, w="zeros")
    x = pa.input_from_history(xd, order)
    print len(x)
    for k in range(0, len(x)):
        p = filt.predict(x[k])
        filt.adapt(s[k], x[k])
        y[k] = p
    return y

def run(df, delays, orders):
    for d in delays:
        for o in orders:
            y, e=predictor(df.O.values, delay=d, order=o)
            df["o{}_d{}".format(o,d)]=y[:-d]
    print df.corr().O

if __name__ == "__main__":
    delays=range(1, 20)
    orders=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]

    df=pd.read_csv('data/DAT_ASCII_EURGBP_M1_201605.csv', names=['Date', 'O'], sep=';', parse_dates=True, usecols=[0,1], index_col=0)

    print len(df)
    y=predictor2(df.O.values, delay=16, order=100)
    y3=predictor3(df.O.values, delay=16, order=100)
    plt.plot(df.O.values, 'r.-', y, 'b.-', y3, 'g.-')
    plt.show()
    exit(0)
    print "1minute data:"
    run(df, delays=range(1, 20), orders=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100])
    print "-----------"

    print "10minute data"
    run(df.resample("10min").asfreq().dropna(), delays=range(1, 20), orders=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100])
    print "-----------"

    print "30minute data"
    run(df.resample("30min").asfreq().dropna(), delays=range(1, 20), orders=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100])
    print "-----------"
