import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
import sklearn.neighbors as sk_neigh

class LinRegLearner:
    def __init__(self):
        pass

    def train(self, x, y):
        self.p = np.polyfit(x, y, 1)
        print "Coefficienti: ", self.p

    def query(self, x):
        return np.polyval(self.p, x)

class KNNLearner:
    def __init__(self, n):
        self.knn = sk_neigh.KNeighborsRegressor(n_neighbors=n)

    def train(self, x, y):
        self.knn.fit(x.reshape(-1, 1), y)

    def query(self, x):
        return self.knn.predict(x.reshape(-1, 1))

if __name__ == "__main__":
    x = np.arange(20)
    y = np.arange(20) + np.random.rand(20) - .5

    lr = LinRegLearner()
    lr.train(x, y)
    y_h = lr.query(x)

    plt.subplot(211)
    plt.plot(x, y, 'ob', x, y_h, '-r')

    lr = KNNLearner(3)
    lr.train(x,y)
    y_h = lr.query(np.arange(100.0)/5)

    plt.subplot(212)
    plt.plot(x, y, 'ob', np.arange(100.0)/5, y_h, 'o-r')
    plt.show()