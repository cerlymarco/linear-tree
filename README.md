# linear-tree
A python library to build Model Trees with Linear Models at the leaves.

## Overview
Linear Model Trees combine the learning ability of Decision Tree with the predictive and explicative power of Linear Models. 
Like in tree-based algorithms, the data are split according to simple decision rules. The goodness of slits is evaluated in gain terms fitting Linear Models in the nodes. This implies that the models in the leaves are linear instead of constant approximations like in classical Decision Trees. 

linear-tree is developed to be fully integrable with scikit-learn. ```LinearTreeRegressor``` and ```LinearTreeClassifier``` are provided as scikit-learn BaseEstimator. They are wrappers that build a decision tree on the data fitting a linear estimator from ```sklearn.linear_model```. All the models available in [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) can be used as linear estimators. 

## Installation
```shell
pip install linear-tree
```
The module depends on NumPy, SciPy and Scikit-Learn (>=0.23.0). Python 3.6 or above is supported.

## Media
- [Linear Tree: the perfect mix of Linear Model and Decision Tree](https://towardsdatascience.com/linear-tree-the-perfect-mix-of-linear-model-and-decision-tree-2eaed21936b7)
- [Model Tree: handle Data Shifts mixing Linear Model and Decision Tree](https://towardsdatascience.com/model-tree-handle-data-shifts-mixing-linear-model-and-decision-tree-facfd642e42b)

## Usage
##### Regression
```python
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=4,
                       n_informative=2, n_targets=1,
                       random_state=0, shuffle=False)
regr = LinearTreeRegressor(base_estimator=LinearRegression())
regr.fit(X, y)
```
##### Classification
```python
from sklearn.linear_model import RidgeClassifier
from lineartree import LinearTreeClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = LinearTreeClassifier(base_estimator=RidgeClassifier())
clf.fit(X, y)
```
More examples in the [notebooks folder](https://github.com/cerlymarco/linear-tree/tree/main/notebooks).

Check the [API Reference](https://github.com/cerlymarco/linear-tree/blob/main/notebooks/README.md) to see the parameter configurations and the available methods.

## Examples
Show the model tree structure:

![plot tree](https://raw.githubusercontent.com/cerlymarco/linear-tree/master/imgs/plot_tree.png)

Linear Tree Regressor at work:

![linear tree regressor](https://raw.githubusercontent.com/cerlymarco/linear-tree/master/imgs/linear_tree_reg.png)

Linear Tree Classifier at work:

![linear tree classifier](https://raw.githubusercontent.com/cerlymarco/linear-tree/master/imgs/linear_tree_class.png)

Extract and examine coefficients at the leaves:

![leaf coefficients](https://raw.githubusercontent.com/cerlymarco/linear-tree/master/imgs/leaf_coefficients.png)
