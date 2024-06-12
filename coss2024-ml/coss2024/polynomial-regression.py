# This example comes from sklearn 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
# The example shows the relationship between model complexity and whether the model is under or 
# over or well fitted into the observed data.
#
# Written by Weiguang Guan
# Sharcnet/Digital Research Alliance of Canada
# May, 2023

# Exercises
#   - Increase number of samples (n_samples)
#   - Choose different degrees (for example, 1, 2, 7, 25)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score


def true_func(X):
    return np.cos(1.5 * np.pi * X)

# Generate simulated training data
np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = true_func(X) + np.random.randn(n_samples) * 0.1

degrees = [1, 4, 15]

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)

    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)

    linear_regression = LinearRegression()     # without regularization

    # Linear regression with regularization (Ridge, Lasso, ElasticNet)
    #linear_regression = Ridge(alpha=1.0e-6)    # with regularization (1.0e-18, 1.0e-6, 1.0)
    #linear_regression = Lasso(alpha=1.0e-4, max_iter=50000)
    #linear_regression = ElasticNet(alpha=1.0e-4, max_iter=50000)

    pipeline = Pipeline(
        [
            ("Polynomial features", polynomial_features),
            ("Linear regression", linear_regression),
        ]
    )
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(
        pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10
    )
    print("Degree=", degrees[i], " coef=", linear_regression.coef_, " intercept=", linear_regression.intercept_)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_func(X_test), label="True function")
    plt.scatter(X, y, edgecolor="b", s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title(
        "Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
            degrees[i], -scores.mean(), scores.std()
        )
    )
plt.show()
