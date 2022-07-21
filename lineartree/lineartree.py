import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

from ._classes import _predict_branch
from ._classes import _LinearTree, _LinearBoosting, _LinearForest


class LinearTreeRegressor(_LinearTree, RegressorMixin):
    """A Linear Tree Regressor.

    A Linear Tree Regressor is a meta-estimator that combine the learning
    ability of Decision Tree and the predictive power of Linear Models.
    Like in tree-based algorithms, the received data are splitted according
    simple decision rules. The goodness of slits is evaluated in gain terms
    fitting linear models in each node. This implies that the models in the
    leaves are linear instead of constant approximations like in classical
    Decision Tree.

    Parameters
    ----------
    base_estimator : object
        The base estimator to fit on dataset splits.
        The base estimator must be a sklearn.linear_model.

    criterion : {"mse", "rmse", "mae", "poisson"}, default="mse"
        The function to measure the quality of a split. "poisson"
        requires ``y >= 0``.

    max_depth : int, default=5
        The maximum depth of the tree considering only the splitting nodes.
        A higher value implies a higher training time.

    min_samples_split : int or float, default=6
        The minimum number of samples required to split an internal node.
        The minimum valid number of samples in each node is 6.
        A lower value implies a higher training time.
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=0.1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least `min_samples_leaf` training samples in each of the left and
        right branches.
        The minimum valid number of samples in each leaf is 3.
        A lower value implies a higher training time.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    max_bins : int, default=25
        The maximum number of bins to use to search the optimal split in each
        feature. Features with a small number of unique values may use less than
        ``max_bins`` bins. Must be lower than 120 and larger than 10.
        A higher value implies a higher training time.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    categorical_features : int or array-like of int, default=None
        Indicates the categorical features.
        All categorical indices must be in `[0, n_features)`.
        Categorical features are used for splits but are not used in
        model fitting.
        More categorical features imply a higher training time.
        - None : no feature will be considered categorical.
        - integer array-like : integer indices indicating categorical
          features.
        - integer : integer index indicating a categorical
          feature.

    split_features : int or array-like of int, default=None
        Defines which features can be used to split on.
        All split feature indices must be in `[0, n_features)`.
        - None : All features will be used for splitting.
        - integer array-like : integer indices indicating splitting features.
        - integer : integer index indicating a single splitting feature.

    linear_features : int or array-like of int, default=None
        Defines which features are used for the linear model in the leaves.
        All linear feature indices must be in `[0, n_features)`.
        - None : All features except those in `categorical_features`
          will be used in the leaf models.
        - integer array-like : integer indices indicating features to
          be used in the leaf models.
        - integer : integer index indicating a single feature to be used
          in the leaf models.

    n_jobs : int, default=None
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    feature_importances_ : ndarray of shape (n_features, )
        Normalized total reduction of criteria by splitting features.

    n_targets_ : int
        The number of targets when :meth:`fit` is performed.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from lineartree import LinearTreeRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4,
    ...                        n_informative=2, n_targets=1,
    ...                        random_state=0, shuffle=False)
    >>> regr = LinearTreeRegressor(base_estimator=LinearRegression())
    >>> regr.fit(X, y)
    >>> regr.predict([[0, 0, 0, 0]])
    array([8.8817842e-16])
    """
    def __init__(self, base_estimator, *, criterion='mse', max_depth=5,
                 min_samples_split=6, min_samples_leaf=0.1, max_bins=25,
                 min_impurity_decrease=0.0, categorical_features=None,
                 split_features=None, linear_features=None, n_jobs=None):

        self.base_estimator = base_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.min_impurity_decrease = min_impurity_decrease
        self.categorical_features = categorical_features
        self.split_features = split_features
        self.linear_features = linear_features
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Build a Linear Tree of a linear estimator from the training
        set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, ) or (n_samples, n_targets)
            Target values.

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that if the base estimator does not support sample weighting,
            the sample weights are still used to evaluate the splits.

        Returns
        -------
        self : object
        """
        reg_criterions = ('mse', 'rmse', 'mae', 'poisson')

        if self.criterion not in reg_criterions:
            raise ValueError("Regression tasks support only criterion in {}, "
                             "got '{}'.".format(reg_criterions, self.criterion))

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y,
            reset=True,
            accept_sparse=False,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=True,
            y_numeric=True,
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        y_shape = np.shape(y)
        self.n_targets_ = y_shape[1] if len(y_shape) > 1 else 1
        if self.n_targets_ < 2:
            y = y.ravel()
        self._fit(X, y, sample_weight)

        return self

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, ) or also (n_samples, n_targets) if
            multitarget regression.
            The predicted values.
        """
        check_is_fitted(self, attributes='_nodes')

        X = self._validate_data(
            X,
            reset=False,
            accept_sparse=False,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_features=self.n_features_in_
        )

        if self.n_targets_ > 1:
            pred = np.zeros((X.shape[0], self.n_targets_))
        else:
            pred = np.zeros(X.shape[0])

        for L in self._leaves.values():

            mask = _predict_branch(X, L.threshold)
            if (~mask).all():
                continue

            pred[mask] = L.model.predict(X[np.ix_(mask, self._linear_features)])

        return pred


class LinearTreeClassifier(_LinearTree, ClassifierMixin):
    """A Linear Tree Classifier.

    A Linear Tree Classifier is a meta-estimator that combine the learning
    ability of Decision Tree and the predictive power of Linear Models.
    Like in tree-based algorithms, the received data are splitted according
    simple decision rules. The goodness of slits is evaluated in gain terms
    fitting linear models in each node. This implies that the models in the
    leaves are linear instead of constant approximations like in classical
    Decision Tree.

    Parameters
    ----------
    base_estimator : object
        The base estimator to fit on dataset splits.
        The base estimator must be a sklearn.linear_model.
        The selected base estimator is automatically substituted by a
        `~sklearn.dummy.DummyClassifier` when a dataset split
        is composed of unique labels.

    criterion : {"hamming", "crossentropy"}, default="hamming"
        The function to measure the quality of a split. `"crossentropy"`
        can be used only if `base_estimator` has `predict_proba` method.

    max_depth : int, default=5
        The maximum depth of the tree considering only the splitting nodes.
        A higher value implies a higher training time.

    min_samples_split : int or float, default=6
        The minimum number of samples required to split an internal node.
        The minimum valid number of samples in each node is 6.
        A lower value implies a higher training time.
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=0.1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least `min_samples_leaf` training samples in each of the left and
        right branches.
        The minimum valid number of samples in each leaf is 3.
        A lower value implies a higher training time.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    max_bins : int, default=25
        The maximum number of bins to use to search the optimal split in each
        feature. Features with a small number of unique values may use less than
        ``max_bins`` bins. Must be lower than 120 and larger than 10.
        A higher value implies a higher training time.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    categorical_features : int or array-like of int, default=None
        Indicates the categorical features.
        All categorical indices must be in `[0, n_features)`.
        Categorical features are used for splits but are not used in
        model fitting.
        More categorical features imply a higher training time.
        - None : no feature will be considered categorical.
        - integer array-like : integer indices indicating categorical
          features.
        - integer : integer index indicating a categorical
          feature.

    split_features : int or array-like of int, default=None
        Defines which features can be used to split on.
        All split feature indices must be in `[0, n_features)`.
        - None : All features will be used for splitting.
        - integer array-like : integer indices indicating splitting features.
        - integer : integer index indicating a single splitting feature.

    linear_features : int or array-like of int, default=None
        Defines which features are used for the linear model in the leaves.
        All linear feature indices must be in `[0, n_features)`.
        - None : All features except those in `categorical_features`
          will be used in the leaf models.
        - integer array-like : integer indices indicating features to
          be used in the leaf models.
        - integer : integer index indicating a single feature to be used
          in the leaf models.

    n_jobs : int, default=None
        The number of jobs to run in parallel for model fitting.
        ``None`` means 1 using one processor. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    feature_importances_ : ndarray of shape (n_features, )
        Normalized total reduction of criteria by splitting features.

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from lineartree import LinearTreeClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = LinearTreeClassifier(base_estimator=RidgeClassifier())
    >>> clf.fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    """
    def __init__(self, base_estimator, *, criterion='hamming', max_depth=5,
                 min_samples_split=6, min_samples_leaf=0.1, max_bins=25,
                 min_impurity_decrease=0.0, categorical_features=None,
                 split_features=None, linear_features=None, n_jobs=None):

        self.base_estimator = base_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.min_impurity_decrease = min_impurity_decrease
        self.categorical_features = categorical_features
        self.split_features = split_features
        self.linear_features = linear_features
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Build a Linear Tree of a linear estimator from the training
        set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, )
            Target values.

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that if the base estimator does not support sample weighting,
            the sample weights are still used to evaluate the splits.

        Returns
        -------
        self : object
        """
        clas_criterions = ('hamming', 'crossentropy')

        if self.criterion not in clas_criterions:
            raise ValueError("Classification tasks support only criterion in {}, "
                             "got '{}'.".format(clas_criterions, self.criterion))

        if (not hasattr(self.base_estimator, 'predict_proba') and
                self.criterion == 'crossentropy'):
            raise ValueError("The 'crossentropy' criterion requires a base_estimator "
                             "with predict_proba method.")

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y,
            reset=True,
            accept_sparse=False,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=False,
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.classes_ = np.unique(y)
        self._fit(X, y, sample_weight)

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, )
            The predicted classes.
        """
        check_is_fitted(self, attributes='_nodes')

        X = self._validate_data(
            X,
            reset=False,
            accept_sparse=False,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_features=self.n_features_in_
        )

        pred = np.empty(X.shape[0], dtype=self.classes_.dtype)

        for L in self._leaves.values():

            mask = _predict_branch(X, L.threshold)
            if (~mask).all():
                continue

            pred[mask] = L.model.predict(X[np.ix_(mask, self._linear_features)])

        return pred

    def predict_proba(self, X):
        """Predict class probabilities for X.

        If base estimators do not implement a ``predict_proba`` method,
        then the one-hot encoding of the predicted class is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self, attributes='_nodes')

        X = self._validate_data(
            X,
            reset=False,
            accept_sparse=False,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_features=self.n_features_in_
        )

        pred = np.zeros((X.shape[0], len(self.classes_)))

        if hasattr(self.base_estimator, 'predict_proba'):
            for L in self._leaves.values():

                mask = _predict_branch(X, L.threshold)
                if (~mask).all():
                    continue

                pred[np.ix_(mask, np.isin(self.classes_, L.classes))] = \
                    L.model.predict_proba(X[np.ix_(mask, self._linear_features)])

        else:
            pred_class = self.predict(X)
            class_to_int = dict(map(reversed, enumerate(self.classes_)))
            pred_class = np.array([class_to_int[i] for i in pred_class])
            pred[np.arange(X.shape[0]), pred_class] = 1

        return pred

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        If base estimators do not implement a ``predict_log_proba`` method,
        then the logarithm of the one-hot encoded predicted class is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        return np.log(self.predict_proba(X))


class LinearBoostRegressor(_LinearBoosting, RegressorMixin):
    """A Linear Boosting Regressor.

    A Linear Boosting Regressor is an iterative meta-estimator that starts
    with a linear regressor, and model the residuals through decision trees.
    At each iteration, the path leading to highest error (i.e. the worst leaf)
    is added as a new binary variable to the base model. This kind of Linear
    Boosting can be considered as an improvement over general linear models
    since it enables incorporating non-linear features by residuals modeling.

    Parameters
    ----------
    base_estimator : object
        The base estimator iteratively fitted.
        The base estimator must be a sklearn.linear_model.

    loss : {"linear", "square", "absolute", "exponential"}, default="linear"
        The function used to calculate the residuals of each sample.

    n_estimators : int, default=10
        The number of boosting stages to perform. It corresponds to the number
        of the new features generated.

    max_depth : int, default=3
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    n_features_out_ : int
        The total number of features used to fit the base estimator in the
        last iteration. The number of output features is equal to the sum
        of n_features_in_ and n_estimators.

    coef_ : array of shape (n_features_out_, ) or (n_targets, n_features_out_)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this is a
        2D array of shape (n_targets, n_features_out_), while if only one target
        is passed, this is a 1D array of length n_features_out_.

    intercept_ : float or array of shape (n_targets, )
        Independent term in the linear model. Set to 0 if `fit_intercept = False`
        in `base_estimator`

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from lineartree import LinearBoostRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4,
    ...                        n_informative=2, n_targets=1,
    ...                        random_state=0, shuffle=False)
    >>> regr = LinearBoostRegressor(base_estimator=LinearRegression())
    >>> regr.fit(X, y)
    >>> regr.predict([[0, 0, 0, 0]])
    array([8.8817842e-16])

    References
    ----------
    Explainable boosted linear regression for time series forecasting.
    Authors: Igor Ilic, Berk Gorgulu, Mucahit Cevik, Mustafa Gokce Baydogan.
    (https://arxiv.org/abs/2009.09110)
    """
    def __init__(self, base_estimator, *, loss='linear', n_estimators=10,
                 max_depth=3, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, ccp_alpha=0.0):

        self.base_estimator = base_estimator
        self.loss = loss
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        """Build a Linear Boosting from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, ) or (n_samples, n_targets)
            Target values.

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights.

        Returns
        -------
        self : object
        """
        reg_losses = ('linear', 'square', 'absolute', 'exponential')

        if self.loss not in reg_losses:
            raise ValueError("Regression tasks support only loss in {}, "
                             "got '{}'.".format(reg_losses, self.loss))

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y,
            reset=True,
            accept_sparse=False,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=True,
            y_numeric=True,
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        y_shape = np.shape(y)
        n_targets = y_shape[1] if len(y_shape) > 1 else 1
        if n_targets < 2:
            y = y.ravel()
        self._fit(X, y, sample_weight)

        return self

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, ) or also (n_samples, n_targets) if
            multitarget regression.
            The predicted values.
        """
        check_is_fitted(self, attributes='base_estimator_')

        return self.base_estimator_.predict(self.transform(X))


class LinearBoostClassifier(_LinearBoosting, ClassifierMixin):
    """A Linear Boosting Classifier.

    A Linear Boosting Classifier is an iterative meta-estimator that starts
    with a linear classifier, and model the residuals through decision trees.
    At each iteration, the path leading to highest error (i.e. the worst leaf)
    is added as a new binary variable to the base model. This kind of Linear
    Boosting can be considered as an improvement over general linear models
    since it enables incorporating non-linear features by residuals modeling.

    Parameters
    ----------
    base_estimator : object
        The base estimator iteratively fitted.
        The base estimator must be a sklearn.linear_model.

    loss : {"hamming", "entropy"}, default="entropy"
        The function used to calculate the residuals of each sample.
        `"entropy"` can be used only if `base_estimator` has `predict_proba`
        method.

    n_estimators : int, default=10
        The number of boosting stages to perform. It corresponds to the number
        of the new features generated.

    max_depth : int, default=3
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    n_features_out_ : int
        The total number of features used to fit the base estimator in the
        last iteration. The number of output features is equal to the sum
        of n_features_in_ and n_estimators.

    coef_ : ndarray of shape (1, n_features_out_) or (n_classes, n_features_out_)
        Coefficient of the features in the decision function.

    intercept_ : float or array of shape (n_classes, )
        Independent term in the linear model. Set to 0 if `fit_intercept = False`
        in `base_estimator`

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from lineartree import LinearBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = LinearBoostClassifier(base_estimator=RidgeClassifier())
    >>> clf.fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])

    References
    ----------
    Explainable boosted linear regression for time series forecasting.
    Authors: Igor Ilic, Berk Gorgulu, Mucahit Cevik, Mustafa Gokce Baydogan.
    (https://arxiv.org/abs/2009.09110)
    """
    def __init__(self, base_estimator, *, loss='hamming', n_estimators=10,
                 max_depth=3, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, ccp_alpha=0.0):

        self.base_estimator = base_estimator
        self.loss = loss
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        """Build a Linear Boosting from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, )
            Target values.

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights.

        Returns
        -------
        self : object
        """
        clas_losses = ('hamming', 'entropy')

        if self.loss not in clas_losses:
            raise ValueError("Classification tasks support only loss in {}, "
                             "got '{}'.".format(clas_losses, self.loss))

        if (not hasattr(self.base_estimator, 'predict_proba') and
                self.loss == 'entropy'):
            raise ValueError("The 'entropy' loss requires a base_estimator "
                             "with predict_proba method.")

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y,
            reset=True,
            accept_sparse=False,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=False,
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.classes_ = np.unique(y)
        self._fit(X, y, sample_weight)

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, )
            The predicted classes.
        """
        check_is_fitted(self, attributes='base_estimator_')

        return self.base_estimator_.predict(self.transform(X))

    def predict_proba(self, X):
        """Predict class probabilities for X.

        If base estimators do not implement a ``predict_proba`` method,
        then the one-hot encoding of the predicted class is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        if hasattr(self.base_estimator, 'predict_proba'):
            check_is_fitted(self, attributes='base_estimator_')
            pred = self.base_estimator_.predict_proba(self.transform(X))

        else:
            pred_class = self.predict(X)
            pred = np.zeros((pred_class.shape[0], len(self.classes_)))
            class_to_int = dict(map(reversed, enumerate(self.classes_)))
            pred_class = np.array([class_to_int[v] for v in pred_class])
            pred[np.arange(pred_class.shape[0]), pred_class] = 1

        return pred

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        If base estimators do not implement a ``predict_log_proba`` method,
        then the logarithm of the one-hot encoded predicted class is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        return np.log(self.predict_proba(X))


class LinearForestRegressor(_LinearForest, RegressorMixin):
    """"A Linear Forest Regressor.

    Linear forests generalizes the well known random forests by combining
    linear models with the same random forests.
    The key idea of linear forests is to use the strength of linear models
    to improve the nonparametric learning ability of tree-based algorithms.
    Firstly, a linear model is fitted on the whole dataset, then a random
    forest is trained on the same dataset but using the residuals of the
    previous steps as target. The final predictions are the sum of the raw
    linear predictions and the residuals modeled by the random forest.

    Parameters
    ----------
    base_estimator : object
        The linear estimator fitted on the raw target.
        The linear estimator must be a regressor from sklearn.linear_model.

    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1]`.

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    feature_importances_ : ndarray of shape (n_features, )
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this is a
        2D array of shape (n_targets, n_features), while if only one target
        is passed, this is a 1D array of length n_features.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0 if `fit_intercept = False`
        in `base_estimator`.

    base_estimator_ : object
        A fitted linear model instance.

    forest_estimator_ : object
        A fitted random forest instance.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from lineartree import LinearForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4,
    ...                        n_informative=2, n_targets=1,
    ...                        random_state=0, shuffle=False)
    >>> regr = LinearForestRegressor(base_estimator=LinearRegression())
    >>> regr.fit(X, y)
    >>> regr.predict([[0, 0, 0, 0]])
    array([8.8817842e-16])

    References
    ----------
    Regression-Enhanced Random Forests.
    Authors: Haozhe Zhang, Dan Nettleton, Zhengyuan Zhu.
    (https://arxiv.org/abs/1904.10416)
    """
    def __init__(self, base_estimator, *, n_estimators=100,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto",
                 max_leaf_nodes=None, min_impurity_decrease=0.,
                 bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, ccp_alpha=0.0, max_samples=None):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

    def fit(self, X, y, sample_weight=None):
        """Build a Linear Forest from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, ) or (n_samples, n_targets)
            Target values.

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights.

        Returns
        -------
        self : object
        """
        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y,
            reset=True,
            accept_sparse=True,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=True,
            y_numeric=True,
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        y_shape = np.shape(y)
        n_targets = y_shape[1] if len(y_shape) > 1 else 1
        if n_targets < 2:
            y = y.ravel()
        self._fit(X, y, sample_weight)

        return self

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, ) or also (n_samples, n_targets) if
            multitarget regression.
            The predicted values.
        """
        check_is_fitted(self, attributes='base_estimator_')

        X = self._validate_data(
            X,
            reset=False,
            accept_sparse=True,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_features=self.n_features_in_
        )

        linear_pred = self.base_estimator_.predict(X)
        forest_pred = self.forest_estimator_.predict(X)

        return linear_pred + forest_pred


class LinearForestClassifier(_LinearForest, ClassifierMixin):
    """"A Linear Forest Classifier.

    Linear forests generalizes the well known random forests by combining
    linear models with the same random forests.
    The key idea of linear forests is to use the strength of linear models
    to improve the nonparametric learning ability of tree-based algorithms.
    Firstly, a linear model is fitted on the whole dataset, then a random
    forest is trained on the same dataset but using the residuals of the
    previous steps as target. The final predictions are the sum of the raw
    linear predictions and the residuals modeled by the random forest.

    For classification tasks the same approach used in regression context
    is adopted. The binary targets are transformed into logits using the
    inverse sigmoid function. A linear regression is fitted. A random forest
    regressor is trained to approximate the residulas from logits and linear
    predictions. Finally the sigmoid of the combinded predictions are taken
    to obtain probabilities.
    The multi-label scenario is carried out using OneVsRestClassifier.

    Parameters
    ----------
    base_estimator : object
        The linear estimator fitted on the raw target.
        The linear estimator must be a regressor from sklearn.linear_model.

    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1]`.

    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.

    feature_importances_ : ndarray of shape (n_features, )
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

    coef_ : ndarray of shape (1, n_features_out_)
        Coefficient of the features in the decision function.

    intercept_ : float
        Independent term in the linear model. Set to 0 if `fit_intercept = False`
        in `base_estimator`.

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    base_estimator_ : object
        A fitted linear model instance.

    forest_estimator_ : object
        A fitted random forest instance.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from lineartree import LinearForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_classes=2, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = LinearForestClassifier(base_estimator=LinearRegression())
    >>> clf.fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])

    References
    ----------
    Regression-Enhanced Random Forests.
    Authors: Haozhe Zhang, Dan Nettleton, Zhengyuan Zhu.
    (https://arxiv.org/abs/1904.10416)
    """
    def __init__(self, base_estimator, *, n_estimators=100,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto",
                 max_leaf_nodes=None, min_impurity_decrease=0.,
                 bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, ccp_alpha=0.0, max_samples=None):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

    def fit(self, X, y, sample_weight=None):
        """Build a Linear Forest from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, ) or (n_samples, n_targets)
            Target values.

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights.

        Returns
        -------
        self : object
        """
        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y,
            reset=True,
            accept_sparse=True,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=False,
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError(
                "LinearForestClassifier supports only binary classification task. "
                "To solve a multi-lable classification task use "
                "LinearForestClassifier with OneVsRestClassifier from sklearn.")

        self._fit(X, y, sample_weight)

        return self

    def decision_function(self, X):
        """Predict confidence scores for samples.

        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, )
            Confidence scores.
            Confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        check_is_fitted(self, attributes='base_estimator_')

        X = self._validate_data(
            X,
            reset=False,
            accept_sparse=True,
            dtype='float32',
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_features=self.n_features_in_
        )

        linear_pred = self.base_estimator_.predict(X)
        forest_pred = self.forest_estimator_.predict(X)

        return linear_pred + forest_pred

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, )
            The predicted classes.
        """
        pred = self.decision_function(X)
        pred_class = (self._sigmoid(pred) > 0.5).astype(int)
        int_to_class = dict(enumerate(self.classes_))
        pred_class = np.array([int_to_class[i] for i in pred_class])

        return pred_class

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """

        pred = self._sigmoid(self.decision_function(X))
        proba = np.zeros((X.shape[0], 2))
        proba[:, 0] = 1 - pred
        proba[:, 1] = pred

        return proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        return np.log(self.predict_proba(X))
