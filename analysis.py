import csv
import datetime as dt
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
import math
import numpy

def bk(k, f_dn, f_up):
    a = 2 * math.pi / f_up
    b = 2 * math.pi / f_dn
    print 'a: ', a, ', b: ', b
    ak_z = [(b-a)/math.pi]
    ak_1k= [ (math.sin(i*b) - math.sin(i*a))/i/math.pi for i in range(1, k)]
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

# Prova a fare una predizione per evitare lo sfasamento:
#i dati futuri sono tutti uguali al tempo k
def bk_filter_2(x, ak):
    k = (len(ak)-1)/2
    y = [0 for i in range(k)]
    y += [numpy.sum(numpy.array(x[i-k:i+1])*numpy.array(ak[0:k+1])) + numpy.sum(numpy.array(ak[k+1:]))*x[i] for i in range(k, len(x))]
    #y += [0 for i in range(k)]
    return y


bid={}
ask={}
#data_file='/home/rek/shared/TickData_EURGBPecn_2016919_1037.csv'
#data_file='/home/rek/shared/TickData_EURGBPecn_2016919_1428.csv'
#data_file='/home/rek/shared/TickData_EURGBPecn_2016927_1915.csv'
data_file='/home/rek/shared/TickData_EURGBPecn_2016929_1740.csv'
with open(data_file, 'r') as csvfile:
	csvdata = csv.DictReader(csvfile, fieldnames=['time', 'bid', 'ask'])
	for r in csvdata:
		t = int(r['time'])
		if t in bid.keys():
			bid[t].append(float(r['bid']))
			ask[t].append(float(r['ask']))
		else:
			bid[t]=[float(r['bid'])]
			ask[t]=[float(r['ask'])]

time = sorted(bid.keys())
bid_v = [sum(bid[t])/len(bid[t]) for t in time]
ask_v = [sum(ask[t])/len(ask[t]) for t in time]

spread_v = [(ask_v[i]-bid_v[i])*10000 for i in range(0,len(bid_v))]
c120=bk(120,3600,36000)
c480=bk(480,3600,3600000)
bid_f_120=bk_filter(bid_v,c120)
bid_f_480=bk_filter(bid_v,c480)
delta=bid_v[600]-bid_f_120[600]
bid_c_120 = map(lambda x: x+delta, bid_f_120)
delta=bid_v[600]-bid_f_480[600]
bid_c_480 = map(lambda x: x+delta, bid_f_480)
plt.subplot(211)
plt.plot(bid_v[500:-500], 'r', bid_c_120[500:-500], 'g', bid_c_480[500:-500], 'b')
#Shift a destra della dimensione del campione, per valutare lo sfasamento
plt.subplot(212)
bid_s_c_120 = bid_c_120[-120:]+bid_c_120[:-120]
bid_s_c_480 = bid_c_480[-480:]+bid_c_480[:-480]
plt.plot(bid_v[1000:], 'r', bid_s_c_120[1000:], 'g', bid_s_c_480[1000:], 'b')
plt.show()


def simulate(actual, filtered, fign):
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
            elif cur > prev + .0005:
                # min found, go for long position
                price = (actual[i]+.0003)*1000
                print "Buying on ", i, " at (bid + 3pip spread) ", actual[i]+.0003, " (", actual[i], ")"
                cur = prev
                order_buy_time.append(i)
                order_buy_price.append(actual[i]+.0003)
        else:
            #order placed
            if cur > prev:
                # going up
                prev = cur
            elif cur < prev - .0005:
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

simulate(bid_v[1000:], bid_s_c_480[1000:], 1)
simulate(bid_v[1000:], bid_s_c_120[1000:], 2)


t=range(2560)
f4=map(lambda x: math.sin(2*math.pi/4*x), t)
f8=map(lambda x: math.sin(2*math.pi/8*x), t)
f16=map(lambda x: math.sin(2*math.pi/16*x), t)
f32=map(lambda x: math.sin(2*math.pi/32*x), t)
f64=map(lambda x: math.sin(2*math.pi/64*x), t)
f128=map(lambda x: math.sin(2*math.pi/128*x), t)
f256=map(lambda x: math.sin(2*math.pi/256*x), t)
f3=map(lambda x: math.sin(2*math.pi/3*x), t)
f17=map(lambda x: math.sin(2*math.pi/17*x), t)
f31=map(lambda x: math.sin(2*math.pi/31*x), t)
f100=map(lambda x: math.sin(2*math.pi/100*x), t)
f145=map(lambda x: math.sin(2*math.pi/145*x), t)
f196=map(lambda x: math.sin(2*math.pi/196*x), t)
f307=map(lambda x: math.sin(2*math.pi/307*x), t)
x=np.array(f4)+np.array(f8)+np.array(f16)+np.array(f32)+np.array(f64)+np.array(f128)+np.array(f256)
x2=np.array(f3)+np.array(f17)+np.array(f31)+np.array(f100)+np.array(f145)+np.array(f196)+np.array(f307)
coeff=bk(48, 50, 100)
y=bk_filter(x, coeff)
plt.plot(t,x, t, y, 'r', t, f64, 'g')
plt.show()
