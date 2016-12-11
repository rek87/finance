import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt

class LinRegLearner:
    def __init__(self):
        pass

    def train(self, x, y):
        self.p = np.polyfit(x, y, 1)
        print "Coefficienti: ", self.p

    def query(self, x):
        return np.polyval(self.p, x)

if __name__ == "__main__":
    x = np.arange(20)
    y = np.arange(20) + np.random.rand(20) - .5

    lr = LinRegLearner()
    lr.train(x, y)
    y_h = lr.query(x)

    plt.plot(x, y, 'ob', x, y_h, '-r')
    plt.show()
