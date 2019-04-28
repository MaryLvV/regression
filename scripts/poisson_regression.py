import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt

# %matplotlib inline

df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/poisson_sim.csv")
X = df[["math"]]
y = df["num_awards"].values

# first look
plt.scatter(X, y)

# vanilla linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression(fit_intercept=True)
lr.fit(X, y)
lr.coef_
lr.intercept_

# new slices of data
from itertools import product


def expand_grid(dictionary):
    return pd.DataFrame(
        [row for row in product(*dictionary.values())], columns=dictionary.keys()
    )


dictionary = {"math": range(30, 80)}
X_new = expand_grid(dictionary)

# linear regression prediction
y_lr = lr.predict(X_new)

plt.scatter(X, y)
plt.plot(X_new, y_lr, c="red")

# quickly find the mean and variance of the data
np.var(y)
np.mean(y)

# Set up Xi and try to do vanilla statsmodels
Xi = sm.add_constant(X)
glm_poi = sm.GLM(y, Xi, family=sm.families.Poisson())
glm_poi = glm_poi.fit()
glm_poi.predict(Xi)
glm_poi.summary()
np.exp(0.0862)

y_sm = glm_poi.predict(sm.add_constant(X_new))

plt.scatter(X, y)
plt.plot(X_new, y_sm, c="red")

# try to use sklearn
# https://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator


class PoissonRegression(BaseEstimator):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        X, y = check_X_y(X, y, accept_sparse=True)
        self._model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        if self.fit_intercept:
            self.coef_ = self._model.params[1:]
            self.intercept_ = self._model.params[0]
        else:
            self.coef_ = self._model.params
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        return self._model.predict(X)


pr = PoissonRegression()
pr.fit(X, y)
pr.predict(X)[:10]

y_pr = pr.predict(X_new)

plt.scatter(X, y, alpha=1 / 4)
plt.plot(X_new, y_lr, c="red")
plt.plot(X_new, y_pr, c="k")

model.coef_
model.intercept_
