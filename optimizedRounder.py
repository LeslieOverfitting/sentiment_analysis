from functools import partial
import numpy as np
import scipy as sp
from sklearn import metrics

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _f1_loss(self, coef, X, y):
        predict = X * coef
        predict_y = np.argmax(predict, axis =1)
        f1_score = metrics.f1_score(y, predict_y, average='macro')  
        return -f1_score * 1000

    def fit(self, X, y):
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [2, 1.0, 2]
        res= sp.optimize.basinhopping(loss_partial, initial_coef, niter=1000)
        print(res.x)
        print(res.fun)
        self.coef_ = res.x
    def get_coef(self):
        return self.coef_