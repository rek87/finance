import csv
import datetime as dt
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
import math
import numpy
import pandas as pd
import padasip as pa

def bk(k, f_dn, f_up):
    a = 2 * math.pi / f_up
    b = 2 * math.pi / f_dn
    print 'a: ', a, ', b: ', b
    ak_z = [(b-a)/math.pi]
    ak_1k= [ (math.sin(i*b) - math.sin(i*a))/i/math.pi for i in range(1, k+1)]
    ak = ak_z + ak_1k
    theta = (1-(ak_z[0] + 2*sum(ak_1k))) / (2*k + 1)
    print 'theta: ',theta
    ak_c = map(lambda x: x + theta, ak)
    ak_f = list(reversed(ak_c[1:])) + [ak_c[0]] + ak_c[1:]
    print "sum(ak): ", sum(ak_f)
    return ak_f

def bk_low(k, f_dn):
    b = 2 * math.pi / f_dn
    print 'b: ', b
    ak_z = [(b)/math.pi]
    ak_1k= [ (math.sin(i*b))/i/math.pi for i in range(1, k+1)]
    ak = ak_z + ak_1k
    theta = (1-(ak_z[0] + 2*sum(ak_1k))) / (2*k + 1)
    print 'theta: ',theta
    ak_c = map(lambda x: x + theta, ak)
    ak_f = list(reversed(ak_c[1:])) + [ak_c[0]] + ak_c[1:]
    print "sum(ak): ", sum(ak_f)
    return ak_f


def bk_filter(x, ak):
    k = (len(ak)-1)/2
    y = [0 for i in range(k)]
    y += [numpy.sum(numpy.array(x[i-k:i+k+1])*numpy.array(ak)) for i in range(k, len(x)-k)]
    y += [0 for i in range(k)]
    return y

# Return a ndarray with filtered data
def np_bkf(x, ak):
    #k = (len(ak)-1)/2
    #f = pd.Series(0, index=x.index)
    #for i in range(k, len(x) - k):
    #    f.iloc[i] = (x.iloc[i-k:i+k+1]['Bid']*ak).sum()
    #return f
    # Full convolution is correct in [(len(ak)-1)/2:-(len(ak)-1)/2] interval
    return numpy.convolve(x, ak)

# Prova a fare una predizione per evitare lo sfasamento:
#i dati futuri sono tutti uguali a 0
def bk_filter_2(x, ak):
    k = (len(ak)-1)/2
    #y = [0 for i in range(k)]
    #y += [numpy.sum(numpy.array(x[i-k:i+1])*numpy.array(ak[0:k+1])*2) for i in range(k, len(x))]
    #y += [0 for i in range(k)]

    aak=numpy.concatenate([2*numpy.array(ak[0:k]),ak[k:k+1]])
    # Coeff needs to be swapped, because they are swapped by convolution
    y = numpy.convolve(x, numpy.flipud(aak))
    return y

# Read data from tick file and return a Pandas DF
def df_from_tick(data_file=None):
    #data_file='/home/rek/shared/TickData_EURGBPecn_2016919_1037.csv'
    #data_file='/home/rek/shared/TickData_EURGBPecn_2016919_1428.csv'
    #data_file='/home/rek/shared/TickData_EURGBPecn_2016927_1915.csv'
    if data_file == None:
        data_file='data/TickData_EURGBPecn_2016929_1740.csv'
    df = pd.read_csv(data_file, names=['Date', 'Bid', 'Ask'], index_col=[0], parse_dates=True)
    # Aggregate data with same index
    df2=df.groupby(df.index).mean()
    return df2

# Adaptation is still wrong
def run_adaptive(s=None, order=3, delay=2):
	if s is None:
		s=numpy.array([numpy.sin(2*numpy.pi*t/100) for t in range(1000)])
	a=numpy.zeros(len(s)+delay)
	filt=pa.filters.FilterNLMS(order, mu=.98)
	for k in range(delay+order-1,len(s)):
		x=s[k-order+1:k+1]  #Input to predictor: last 'order' samples
		a[k+delay]=filt.predict(x)
		#Adapt filter coefficients giving input data who contributed to produce s[k]
		filt.adapt(s[k], s[k-delay-order+1:k-delay+1])
	return a

def run_adaptive2(s=None, order=3, delay=2):
    if s is None:
        s=numpy.array([numpy.sin(2*numpy.pi*t/100) for t in range(1000)])
    a=numpy.zeros(len(s))
    filt=pa.filters.FilterNLMS(order, mu=.98)
    for k in range(order-1,len(s)-delay):
        x=s[k-order:k+1]  #Input to predictor: last 'order' samples
        a[k+delay]=filt.predict(x)
        #Adapt filter coefficients giving input data who contributed to produce s[k]
        filt.adapt(s[k+delay], x)
    return a

def predictor(s=None, delay=2, order=3, mu=.98):
    # Square wave signal
    if s is None:
        s = numpy.zeros(10000)
        for k in range(1, 10, 2):
            s[k*1000:(k+1)*1000] = 1
    y=numpy.zeros(len(s)+delay)
    e=numpy.zeros(len(s))
    filt=pa.filters.FilterNLMS(order, mu=mu)
    for k in range(order+delay-1, len(s)):
        # Calculate y[k-delta] - Useless, is recalculated in adapt method
        p = filt.predict(s[k-delay-order+1:k-delay+1])
        e[k]=s[k]-p
        # Adapt coeff using current x[k] and y[k-delta]
        filt.adapt(s[k], s[k-delay-order+1:k-delay+1])
        y[k+delay] = filt.predict(s[k-order+1:k+1])
    return y, e

if __name__ == "__main__":
    df=pd.read_csv('data/DAT_ASCII_EURGBP_M1_201605.csv', names=['Date', 'O'], sep=';', parse_dates=True, usecols=[0,1], index_col=0)


    c_pad=bk_low(14400, 14400)
    bid_f_pad=bk_filter_2(bid_v,c_pad)
    plt.plot(bid_v, 'b', bid_f_pad, 'g')
    plt.show()
    simulate(bid_v[1500:], bid_f_pad[1500:], 2)

    spread_v = [(ask_v[i]-bid_v[i])*10000 for i in range(0,len(bid_v))]
    c120=bk(120,3600,36000)
    c480=bk(480,3600,3600000)
    bid_f_120=bk_filter(bid_v,c120)
    bid_f_480=bk_filter(bid_v,c480)
    bid_f_480_2=bk_filter_2(bid_v,c480)
    delta=bid_v[600]-bid_f_120[600]
    bid_c_120 = map(lambda x: x+delta, bid_f_120)
    delta=bid_v[600]-bid_f_480[600]
    bid_c_480 = map(lambda x: x+delta, bid_f_480)
    delta=bid_v[600]-bid_f_480_2[600]
    bid_c_480_2 = map(lambda x: x+delta, bid_f_480_2)
    plt.subplot(211)
    plt.plot(bid_v[500:-500], 'r', bid_c_120[500:-500], 'g', bid_c_480[500:-500], 'b', bid_c_480_2[500:-500], 'y')
    #Shift a destra della dimensione del campione, per valutare lo sfasamento
    plt.subplot(212)
    bid_s_c_120 = bid_c_120[-120:]+bid_c_120[:-120]
    bid_s_c_480 = bid_c_480[-480:]+bid_c_480[:-480]
    plt.plot(bid_v[1000:], 'r', bid_s_c_120[1000:], 'g', bid_s_c_480[1000:], 'b')
    plt.show()



def simulate(actual, filtered, fign, band=.0005, spread=.0003):
    prev = filtered[0]
    price = 0
    gain = [0]
    cumul_gain = 0
    order_buy_time=[]
    order_buy_price=[]
    order_sell_time=[]
    order_sell_price=[]
    for i in range(1, len(actual)):
        cur = filtered[i]
        order_gain = 0
        if price == 0:
            # not yet placed an order
            if cur < prev:
                # current value is less than previous, going down
                prev = cur
            elif cur > prev + band:
                # min found, go for long position
                price = (actual[i]+spread)*1000
                print "Buying on ", i, " at (bid + ", spread*10000, "pip spread) ", actual[i]+spread, " (", actual[i], ")"
                cur = prev
                order_buy_time.append(i)
                order_buy_price.append(actual[i]+spread)
        else:
            #order placed
            if cur > prev:
                # going up
                prev = cur
            elif cur < prev - band:
                # max found, sell the order
                order_gain = (actual[i]*1000 - price)
                cumul_gain += order_gain
                print "Selling on ", i, " at ", actual[i], " gained: ", order_gain, " (Tot gain: ", cumul_gain, ")"
                cur = prev
                price = 0
                order_sell_time.append(i)
                order_sell_price.append(actual[i])
        gain.append(gain[i-1] + order_gain)

    plt.figure(fign)
    plt.plot(actual, 'b', filtered, 'g')
    plt.plot(order_buy_time, order_buy_price, 'oy')
    plt.plot(order_sell_time, order_sell_price, 'or')
    plt.show()


#simulate(bid_v[1000:], bid_s_c_480[1000:], 1)
#simulate(bid_v[1000:], bid_s_c_120[1000:], 2)
#
#
#t=range(2560)
#f4=map(lambda x: math.sin(2*math.pi/4*x), t)
#f8=map(lambda x: math.sin(2*math.pi/8*x), t)
#f16=map(lambda x: math.sin(2*math.pi/16*x), t)
#f32=map(lambda x: math.sin(2*math.pi/32*x), t)
#f64=map(lambda x: math.sin(2*math.pi/64*x), t)
#f128=map(lambda x: math.sin(2*math.pi/128*x), t)
#f256=map(lambda x: math.sin(2*math.pi/256*x), t)
#f3=map(lambda x: math.sin(2*math.pi/3*x), t)
#f17=map(lambda x: math.sin(2*math.pi/17*x), t)
#f31=map(lambda x: math.sin(2*math.pi/31*x), t)
#f100=map(lambda x: math.sin(2*math.pi/100*x), t)
#f145=map(lambda x: math.sin(2*math.pi/145*x), t)
#f196=map(lambda x: math.sin(2*math.pi/196*x), t)
#f307=map(lambda x: math.sin(2*math.pi/307*x), t)
#x=np.array(f4)+np.array(f8)+np.array(f16)+np.array(f32)+np.array(f64)+np.array(f128)+np.array(f256)
#x2=np.array(f3)+np.array(f17)+np.array(f31)+np.array(f100)+np.array(f145)+np.array(f196)+np.array(f307)
#coeff=bk(48, 50, 100)
#y=bk_filter(x, coeff)
#plt.plot(t,x, t, y, 'r', t, f64, 'g')
#plt.show()
