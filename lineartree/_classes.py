import numbers
import numpy as np
import scipy.sparse as sp

from copy import deepcopy
from joblib import Parallel, effective_n_jobs  # , delayed

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.base import is_regressor
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

from ._criterion import SCORING
from ._criterion import mse, rmse, mae, poisson
from ._criterion import hamming, crossentropy

import sklearn
_sklearn_v1 = eval(sklearn.__version__.split('.')[0]) > 0


CRITERIA = {"mse": mse,
            "rmse": rmse,
            "mae": mae,
            "poisson": poisson,
            "hamming": hamming,
            "crossentropy": crossentropy}


#########################################################################
### remove when https://github.com/joblib/joblib/issues/1071 is fixed ###
#########################################################################
from sklearn import get_config, config_context
from functools import update_wrapper
import functools

# from sklearn.utils.fixes
def delayed(function):
    """Decorator used to capture the arguments of a function."""
    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs
    return delayed_function

# from sklearn.utils.fixes
class _FuncWrapper:
    """"Load the global configuration before calling the function."""
    def __init__(self, function):
        self.function = function
        self.config = get_config()
        update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        with config_context(**self.config):
            return self.function(*args, **kwargs)
#########################################################################
#########################################################################
#########################################################################


def _partition_columns(columns, n_jobs):
    """Private function to partition columns splitting between jobs."""
    # Compute the number of jobs
    n_columns = len(columns)
    n_jobs = min(effective_n_jobs(n_jobs), n_columns)

    # Partition columns between jobs
    n_columns_per_job = np.full(n_jobs, n_columns // n_jobs, dtype=int)
    n_columns_per_job[:n_columns % n_jobs] += 1
    columns_per_job = np.cumsum(n_columns_per_job)
    columns_per_job = np.split(columns, columns_per_job)
    columns_per_job = columns_per_job[:-1]

    return n_jobs, columns_per_job


def _parallel_binning_fit(split_feat, _self, X, y,
                          weights, support_sample_weight,
                          bins, loss):
    """Private function to find the best column splittings within a job."""
    n_sample, n_feat = X.shape
    feval = CRITERIA[_self.criterion]

    split_t = None
    split_col = None
    left_node = (None, None, None, None)
    right_node = (None, None, None, None)
    largs_left = {'classes': None}
    largs_right = {'classes': None}

    if n_sample < _self._min_samples_split:
        return loss, split_t, split_col, left_node, right_node

    for col, _bin in zip(split_feat, bins):

        for q in _bin:

            # create 1D bool mask for right/left children
            mask = (X[:, col] > q)

            n_left, n_right = (~mask).sum(), mask.sum()

            if n_left < _self._min_samples_leaf or n_right < _self._min_samples_leaf:
                continue

            # create 2D bool mask for right/left children
            left_mesh = np.ix_(~mask, _self._linear_features)
            right_mesh = np.ix_(mask, _self._linear_features)

            model_left = deepcopy(_self.base_estimator)
            model_right = deepcopy(_self.base_estimator)

            if hasattr(_self, 'classes_'):
                largs_left['classes'] = np.unique(y[~mask])
                largs_right['classes'] = np.unique(y[mask])
                if len(largs_left['classes']) == 1:
                    model_left = DummyClassifier(strategy="most_frequent")
                if len(largs_right['classes']) == 1:
                    model_right = DummyClassifier(strategy="most_frequent")

            if weights is None:

                model_left.fit(X[left_mesh], y[~mask])
                loss_left = feval(model_left, X[left_mesh], y[~mask],
                                  **largs_left)
                wloss_left = loss_left * (n_left / n_sample)

                model_right.fit(X[right_mesh], y[mask])
                loss_right = feval(model_right, X[right_mesh], y[mask],
                                   **largs_right)
                wloss_right = loss_right * (n_right / n_sample)

            else:

                if support_sample_weight:

                    model_left.fit(X[left_mesh], y[~mask],
                                   sample_weight=weights[~mask])

                    model_right.fit(X[right_mesh], y[mask],
                                    sample_weight=weights[mask])

                else:

                    model_left.fit(X[left_mesh], y[~mask])

                    model_right.fit(X[right_mesh], y[mask])

                loss_left = feval(model_left, X[left_mesh], y[~mask],
                                  weights=weights[~mask], **largs_left)
                wloss_left = loss_left * (weights[~mask].sum() / weights.sum())

                loss_right = feval(model_right, X[right_mesh], y[mask],
                                   weights=weights[mask], **largs_right)
                wloss_right = loss_right * (weights[mask].sum() / weights.sum())

            total_loss = wloss_left + wloss_right

            # store if best
            if total_loss < loss:
                split_t = q
                split_col = col
                loss = total_loss
                left_node = (model_left, loss_left, wloss_left,
                             n_left, largs_left['classes'])
                right_node = (model_right, loss_right, wloss_right,
                              n_right, largs_right['classes'])

    return loss, split_t, split_col, left_node, right_node


def _map_node(X, feat, direction, split):
    """Utility to map samples to nodes"""
    if direction == 'L':
        mask = (X[:, feat] <= split)
    else:
        mask = (X[:, feat] > split)

    return mask


def _predict_branch(X, branch_history, mask=None):
    """Utility to map samples to branches"""

    if mask is None:
        mask = np.repeat(True, X.shape[0])

    for node in branch_history:
        mask = np.logical_and(_map_node(X, *node), mask)

    return mask


class Node:

    def __init__(self, id=None, threshold=[],
                 parent=None, children=None,
                 n_samples=None, w_loss=None,
                 loss=None, model=None, classes=None):
        self.id = id
        self.threshold = threshold
        self.parent = parent
        self.children = children
        self.n_samples = n_samples
        self.w_loss = w_loss
        self.loss = loss
        self.model = model
        self.classes = classes


class _LinearTree(BaseEstimator):
    """Base class for Linear Tree meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self, base_estimator, *, criterion, max_depth,
                 min_samples_split, min_samples_leaf, max_bins,
                 categorical_features, split_features,
                 linear_features, n_jobs):

        self.base_estimator = base_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.categorical_features = categorical_features
        self.split_features = split_features
        self.linear_features = linear_features
        self.n_jobs = n_jobs

    def _parallel_args(self):
        return {}

    def _split(self, X, y, bins,
               support_sample_weight,
               weights=None,
               loss=None):
        """Evaluate optimal splits in a given node (in a specific partition of
        X and y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, )
            The target values (class labels in classification, real numbers in
            regression).

        bins : array-like of shape (max_bins - 2, )
            The bins to use to find an optimal split. Expressed as percentiles.

        support_sample_weight : bool
            Whether the estimator's fit method supports sample_weight.

        weights : array-like of shape (n_samples, ), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that if the base estimator does not support sample weighting,
            the sample weights are still used to evaluate the splits.

        loss : float, default=None
            The loss of the parent node. A split is computed if the weighted
            loss sum of the two children is lower than the loss of the parent.
            A None value implies the first fit on all the data to evaluate
            the benefits of possible future splits.

        Returns
        -------
        self : object
        """
        # Parallel loops
        n_jobs, split_feat = _partition_columns(self._split_features, self.n_jobs)

        # partition columns splittings between jobs
        all_results = Parallel(n_jobs=n_jobs, verbose=0,
                               **self._parallel_args())(
            delayed(_parallel_binning_fit)(
                feat,
                self, X, y,
                weights, support_sample_weight,
                [bins[i] for i in feat],
                loss
            )
            for feat in split_feat)

        # extract results from parallel loops
        _losses, split_t, split_col = [], [], []
        left_node, right_node = [], []
        for job_res in all_results:
            _losses.append(job_res[0])
            split_t.append(job_res[1])
            split_col.append(job_res[2])
            left_node.append(job_res[3])
            right_node.append(job_res[4])

        # select best results
        _id_best = np.argmin(_losses)
        if _losses[_id_best] < loss:
            split_t = split_t[_id_best]
            split_col = split_col[_id_best]
            left_node = left_node[_id_best]
            right_node = right_node[_id_best]
        else:
            split_t = None
            split_col = None
            left_node = (None, None, None, None, None)
            right_node = (None, None, None, None, None)

        return split_t, split_col, left_node, right_node

    def _grow(self, X, y, weights=None):
        """Grow and prune a Linear Tree from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, )
            The target values (class labels in classification, real numbers in
            regression).

        weights : array-like of shape (n_samples, ), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that if the base estimator does not support sample weighting,
            the sample weights are still used to evaluate the splits.

        Returns
        -------
        self : object
        """
        n_sample, self.n_features_in_ = X.shape
        self.feature_importances_ = np.zeros((self.n_features_in_,))

        # extract quantiles
        bins = np.linspace(0, 1, self.max_bins)[1:-1]
        bins = np.quantile(X, bins, axis=0, interpolation='midpoint')
        bins = list(bins.T)
        bins = [np.unique(X[:, c]) if c in self._categorical_features
                else np.unique(q) for c, q in enumerate(bins)]

        # check if base_estimator supports fitting with sample_weights
        support_sample_weight = has_fit_parameter(self.base_estimator,
                                                  "sample_weight")

        queue = ['']  # queue of the nodes to evaluate for splitting
        # store the results of each node in dicts
        self._nodes = {}
        self._leaves = {}

        # initialize first fit
        largs = {'classes': None}
        model = deepcopy(self.base_estimator)
        if weights is None or not support_sample_weight:
            model.fit(X[:, self._linear_features], y)
        else:
            model.fit(X[:, self._linear_features], y, sample_weight=weights)

        if hasattr(self, 'classes_'):
            largs['classes'] = self.classes_

        loss = CRITERIA[self.criterion](
            model, X[:, self._linear_features], y,
            weights=weights, **largs)

        self._nodes[''] = Node(
            id=0,
            n_samples=n_sample,
            model=model,
            loss=loss,
            classes=largs['classes']
        )

        # in the beginning consider all the samples
        start = np.repeat(True, n_sample)
        mask = start.copy()

        i = 1
        while len(queue) > 0:

            if weights is None:
                split_t, split_col, left_node, right_node = self._split(
                    X[mask], y[mask], bins,
                    support_sample_weight,
                    loss=loss)
            else:
                split_t, split_col, left_node, right_node = self._split(
                    X[mask], y[mask], bins,
                    support_sample_weight, weights[mask],
                    loss=loss)

            # no utility in splitting
            if split_col is None or len(queue[-1]) >= self.max_depth:
                self._leaves[queue[-1]] = self._nodes[queue[-1]]
                del self._nodes[queue[-1]]
                queue.pop()

            else:

                model_left, loss_left, wloss_left, n_left, class_left = \
                    left_node
                model_right, loss_right, wloss_right, n_right, class_right = \
                    right_node
                self.feature_importances_[split_col] += \
                    loss - wloss_left - wloss_right

                self._nodes[queue[-1] + 'L'] = Node(
                    id=i, parent=queue[-1],
                    model=model_left,
                    loss=loss_left,
                    w_loss=wloss_left,
                    n_samples=n_left,
                    threshold=self._nodes[queue[-1]].threshold[:] + [
                        (split_col, 'L', split_t)
                    ]
                )

                self._nodes[queue[-1] + 'R'] = Node(
                    id=i + 1, parent=queue[-1],
                    model=model_right,
                    loss=loss_right,
                    w_loss=wloss_right,
                    n_samples=n_right,
                    threshold=self._nodes[queue[-1]].threshold[:] + [
                        (split_col, 'R', split_t)
                    ]
                )

                if hasattr(self, 'classes_'):
                    self._nodes[queue[-1] + 'L'].classes = class_left
                    self._nodes[queue[-1] + 'R'].classes = class_right

                self._nodes[queue[-1]].children = (queue[-1] + 'L', queue[-1] + 'R')

                i += 2
                q = queue[-1]
                queue.pop()
                queue.extend([q + 'R', q + 'L'])

            if len(queue) > 0:
                loss = self._nodes[queue[-1]].loss
                mask = _predict_branch(
                    X, self._nodes[queue[-1]].threshold, start.copy())

        self.node_count = i

        return self

    def _fit(self, X, y, sample_weight=None):
        """Build a Linear Tree of a linear estimator from the training
        set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, ) or also (n_samples, n_targets) for
            multitarget regression.
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that if the base estimator does not support sample weighting,
            the sample weights are still used to evaluate the splits.

        Returns
        -------
        self : object
        """
        n_sample, n_feat = X.shape

        if isinstance(self.min_samples_split, numbers.Integral):
            if self.min_samples_split < 6:
                raise ValueError(
                    "min_samples_split must be an integer greater than 5 or "
                    "a float in (0.0, 1.0); got the integer {}".format(
                        self.min_samples_split))
            self._min_samples_split = self.min_samples_split
        else:
            if not 0. < self.min_samples_split < 1.:
                raise ValueError(
                    "min_samples_split must be an integer greater than 5 or "
                    "a float in (0.0, 1.0); got the float {}".format(
                        self.min_samples_split))

            self._min_samples_split = int(np.ceil(self.min_samples_split * n_sample))
            self._min_samples_split = max(6, self._min_samples_split)

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if self.min_samples_leaf < 3:
                raise ValueError(
                    "min_samples_leaf must be an integer greater than 2 or "
                    "a float in (0.0, 1.0); got the integer {}".format(
                        self.min_samples_leaf))
            self._min_samples_leaf = self.min_samples_leaf
        else:
            if not 0. < self.min_samples_leaf < 1.:
                raise ValueError(
                    "min_samples_leaf must be an integer greater than 2 or "
                    "a float in (0.0, 1.0); got the float {}".format(
                        self.min_samples_leaf))

            self._min_samples_leaf = int(np.ceil(self.min_samples_leaf * n_sample))
            self._min_samples_leaf = max(3, self._min_samples_leaf)

        if not 1 <= self.max_depth <= 20:
            raise ValueError("max_depth must be an integer in [1, 20].")

        if not 10 <= self.max_bins <= 120:
            raise ValueError("max_bins must be an integer in [10, 120].")

        if not hasattr(self.base_estimator, 'fit_intercept'):
            raise ValueError(
                "Only linear models are accepted as base_estimator. "
                "Select one from linear_model class of scikit-learn.")

        if self.categorical_features is not None:
            cat_features = np.unique(self.categorical_features)

            if not issubclass(cat_features.dtype.type, numbers.Integral):
                raise ValueError(
                    "No valid specification of categorical columns. "
                    "Only a scalar, list or array-like of integers is allowed.")

            if (cat_features < 0).any() or (cat_features >= n_feat).any():
                raise ValueError(
                    'Categorical features must be in [0, {}].'.format(
                        n_feat - 1))

            if len(cat_features) == n_feat:
                raise ValueError(
                    "Only categorical features detected. "
                    "No features available for fitting.")
        else:
            cat_features = []
        self._categorical_features = cat_features

        if self.split_features is not None:
            split_features = np.unique(self.split_features)

            if not issubclass(split_features.dtype.type, numbers.Integral):
                raise ValueError(
                    "No valid specification of split_features. "
                    "Only a scalar, list or array-like of integers is allowed.")

            if (split_features < 0).any() or (split_features >= n_feat).any():
                raise ValueError(
                    'Splitting features must be in [0, {}].'.format(
                        n_feat - 1))
        else:
            split_features = np.arange(n_feat)
        self._split_features = split_features

        if self.linear_features is not None:
            linear_features = np.unique(self.linear_features)

            if not issubclass(linear_features.dtype.type, numbers.Integral):
                raise ValueError(
                    "No valid specification of linear_features. "
                    "Only a scalar, list or array-like of integers is allowed.")

            if (linear_features < 0).any() or (linear_features >= n_feat).any():
                raise ValueError(
                    'Linear features must be in [0, {}].'.format(
                        n_feat - 1))

            if np.isin(linear_features, cat_features).any():
                raise ValueError(
                    "Linear features cannot be categorical features.")
        else:
            linear_features = np.setdiff1d(np.arange(n_feat), cat_features)
        self._linear_features = linear_features

        self._grow(X, y, sample_weight)

        normalizer = np.sum(self.feature_importances_)
        if normalizer > 0:
            self.feature_importances_ /= normalizer

        return self

    def summary(self, feature_names=None, only_leaves=False, max_depth=None):
        """Return a summary of nodes created from model fitting.

        Parameters
        ----------
        feature_names : array-like of shape (n_features, ), default=None
            Names of each of the features. If None, generic names
            will be used (“X[0]”, “X[1]”, …).

        only_leaves : bool, default=False
            Store only information of leaf nodes.

        max_depth : int, default=None
            The maximum depth of the representation. If None, the tree
            is fully generated.

        Returns
        -------
        summary : nested dict
            The keys are the integer map of each node.
            The values are dicts containing information for that node:

                - 'col' (^): column used for splitting;
                - 'th' (^): threshold value used for splitting in the
                  selected column;
                - 'loss': loss computed at node level. Weighted sum of
                  children' losses if it is a splitting node;
                - 'samples': number of samples in the node. Sum of children'
                  samples if it is a split node;
                - 'children' (^): integer mapping of possible children nodes;
                - 'models': fitted linear models built in each split.
                  Single model if it is leaf node;
                - 'classes' (^^): target classes detected in the split.
                  Available only for LinearTreeClassifier.

            (^): Only for split nodes.
            (^^): Only for leaf nodes.
        """
        check_is_fitted(self, attributes='_nodes')

        if max_depth is None:
            max_depth = 20
        if max_depth < 1:
            raise ValueError(
                "max_depth must be > 0, got {}".format(max_depth))

        summary = {}

        if len(self._nodes) > 0 and not only_leaves:

            if (feature_names is not None and
                    len(feature_names) != self.n_features_in_):
                raise ValueError(
                    "feature_names must contain {} elements, got {}".format(
                        self.n_features_in_, len(feature_names)))

            if feature_names is None:
                feature_names = np.arange(self.n_features_in_)

            for n, N in self._nodes.items():

                if len(n) >= max_depth:
                    continue

                cl, cr = N.children
                Cl = (self._nodes[cl] if cl in self._nodes
                      else self._leaves[cl])
                Cr = (self._nodes[cr] if cr in self._nodes
                      else self._leaves[cr])

                summary[N.id] = {
                    'col': feature_names[Cl.threshold[-1][0]],
                    'th': round(Cl.threshold[-1][-1], 4),
                    'loss': round(Cl.w_loss + Cr.w_loss, 4),
                    'samples': Cl.n_samples + Cr.n_samples,
                    'children': (Cl.id, Cr.id),
                    'models': (Cl.model, Cr.model)
                }

        for l, L in self._leaves.items():

            if len(l) > max_depth:
                continue

            summary[L.id] = {
                'loss': round(L.loss, 4),
                'samples': L.n_samples,
                'models': L.model
            }

            if hasattr(self, 'classes_'):
                summary[L.id]['classes'] = L.classes

        return summary

    def apply(self, X):
        """Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, )
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; n_nodes)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self, attributes='_nodes')

        X = check_array(
            X, accept_sparse=False, dtype=None,
            force_all_finite=False)
        self._check_n_features(X, reset=False)

        X_leaves = np.zeros(X.shape[0], dtype='int64')

        for L in self._leaves.values():

            mask = _predict_branch(X, L.threshold)
            if (~mask).all():
                continue

            X_leaves[mask] = L.id

        return X_leaves

    def decision_path(self, X):
        """Return the decision path in the tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        check_is_fitted(self, attributes='_nodes')

        X = check_array(
            X, accept_sparse=False, dtype=None,
            force_all_finite=False)
        self._check_n_features(X, reset=False)

        indicator = np.zeros((X.shape[0], self.node_count), dtype='int64')

        for L in self._leaves.values():

            mask = _predict_branch(X, L.threshold)
            if (~mask).all():
                continue

            n = L.id
            p = L.parent
            paths_id = [n]

            while p is not None:
                n = self._nodes[p].id
                p = self._nodes[p].parent
                paths_id.append(n)

            indicator[np.ix_(mask, paths_id)] = 1

        return sp.csr_matrix(indicator)

    def model_to_dot(self, feature_names=None, max_depth=None):
        """Convert a fitted Linear Tree model to dot format.
        It results in ModuleNotFoundError if graphviz or pydot are not available.
        When installing graphviz make sure to add it to the system path.

        Parameters
        ----------
        feature_names : array-like of shape (n_features, ), default=None
            Names of each of the features. If None, generic names
            will be used (“X[0]”, “X[1]”, …).

        max_depth : int, default=None
            The maximum depth of the representation. If None, the tree
            is fully generated.

        Returns
        -------
        graph : pydot.Dot instance
            Return an instance representing the Linear Tree. Splitting nodes have
            a rectangular shape while leaf nodes have a circular one.
        """
        import pydot

        summary = self.summary(feature_names=feature_names, max_depth=max_depth)
        graph = pydot.Dot('linear_tree', graph_type='graph')

        # create nodes
        for n in summary:
            if 'col' in summary[n]:
                if isinstance(summary[n]['col'], str):
                    msg = "id_node: {}\n{} <= {}\nloss: {:.4f}\nsamples: {}"
                else:
                    msg = "id_node: {}\nX[{}] <= {}\nloss: {:.4f}\nsamples: {}"

                msg = msg.format(
                    n, summary[n]['col'], summary[n]['th'],
                    summary[n]['loss'], summary[n]['samples']
                )
                graph.add_node(pydot.Node(n, label=msg, shape='rectangle'))

                for c in summary[n]['children']:
                    if c not in summary:
                        graph.add_node(pydot.Node(c, label="...",
                                                  shape='rectangle'))

            else:
                msg = "id_node: {}\nloss: {:.4f}\nsamples: {}".format(
                    n, summary[n]['loss'], summary[n]['samples'])
                graph.add_node(pydot.Node(n, label=msg))

        # add edges
        for n in summary:
            if 'children' in summary[n]:
                for c in summary[n]['children']:
                    graph.add_edge(pydot.Edge(n, c))

        return graph

    def plot_model(self, feature_names=None, max_depth=None):
        """Convert a fitted Linear Tree model to dot format and display it.
        It results in ModuleNotFoundError if graphviz or pydot are not available.
        When installing graphviz make sure to add it to the system path.

        Parameters
        ----------
        feature_names : array-like of shape (n_features, ), default=None
            Names of each of the features. If None, generic names
            will be used (“X[0]”, “X[1]”, …).

        max_depth : int, default=None
            The maximum depth of the representation. If None, the tree
            is fully generated.

        Returns
        -------
        A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks.
        Splitting nodes have a rectangular shape while leaf nodes
        have a circular one.
        """
        from IPython.display import Image

        graph = self.model_to_dot(feature_names=feature_names, max_depth=max_depth)

        return Image(graph.create_png())


class _LinearBoosting(TransformerMixin, BaseEstimator):
    """Base class for Linear Boosting meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self, base_estimator, *, loss, n_estimators,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_features,
                 random_state, max_leaf_nodes,
                 min_impurity_decrease, ccp_alpha):

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

    def _fit(self, X, y, sample_weight=None):
        """Build a Linear Boosting from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, ) or also (n_samples, n_targets) for
            multitarget regression.
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights.

        Returns
        -------
        self : object
        """
        if not hasattr(self.base_estimator, 'fit_intercept'):
            raise ValueError("Only linear models are accepted as base_estimator. "
                             "Select one from linear_model class of scikit-learn.")

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be an integer greater than 0 but "
                             "got {}".format(self.n_estimators))

        n_sample, self.n_features_in_ = X.shape

        self._trees = []
        self._leaves = []

        for i in range(self.n_estimators):

            estimator = deepcopy(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weight)

            if self.loss == 'entropy':
                pred = estimator.predict_proba(X)
            else:
                pred = estimator.predict(X)

            if hasattr(self, 'classes_'):
                resid = SCORING[self.loss](y, pred, self.classes_)
            else:
                resid = SCORING[self.loss](y, pred)

            if resid.ndim > 1:
                resid = resid.mean(1)

            criterion = 'squared_error' if _sklearn_v1 else 'mse'

            tree = DecisionTreeRegressor(
                criterion=criterion, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha
            )

            tree.fit(X, resid, sample_weight=sample_weight, check_input=False)
            self._trees.append(tree)

            pred_tree = np.abs(tree.predict(X, check_input=False))
            worst_pred = np.max(pred_tree)
            self._leaves.append(worst_pred)

            pred_tree = (pred_tree == worst_pred).astype(np.float32)
            pred_tree = pred_tree.reshape(-1, 1)
            X = np.concatenate([X, pred_tree], axis=1)

        self.base_estimator_ = deepcopy(self.base_estimator)
        self.base_estimator_.fit(X, y, sample_weight=sample_weight)

        if hasattr(self.base_estimator_, 'coef_'):
            self.coef_ = self.base_estimator_.coef_

        if hasattr(self.base_estimator_, 'intercept_'):
            self.intercept_ = self.base_estimator_.intercept_

        self.n_features_out_ = X.shape[1]

        return self

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_out)
            Transformed dataset.
            `n_out` is equal to `n_features` + `n_estimators`
        """
        check_is_fitted(self, attributes='base_estimator_')
        X = check_array(X, dtype=np.float32, accept_sparse=False)
        self._check_n_features(X, reset=False)

        for tree, leaf in zip(self._trees, self._leaves):
            pred_tree = np.abs(tree.predict(X, check_input=False))
            pred_tree = (pred_tree == leaf).astype(np.float32)
            pred_tree = pred_tree.reshape(-1, 1)
            X = np.concatenate([X, pred_tree], axis=1)

        return X


class _LinearForest(BaseEstimator):
    """Base class for Linear Forest meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self, base_estimator, *, n_estimators, max_depth,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_features, max_leaf_nodes, min_impurity_decrease,
                 bootstrap, oob_score, n_jobs, random_state,
                 ccp_alpha, max_samples):

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

    def _sigmoid(self, y):
        """Expit function (a.k.a. logistic sigmoid).

        Parameters
        ----------
        y : array-like of shape (n_samples, )
            The array to apply expit to element-wise.

        Returns
        -------
        y : array-like of shape (n_samples, )
            Expits.
        """
        return np.exp(y) / (1 + np.exp(y))

    def _inv_sigmoid(self, y):
        """Logit function.

        Parameters
        ----------
        y : array-like of shape (n_samples, )
            The array to apply logit to element-wise.

        Returns
        -------
        y : array-like of shape (n_samples, )
            Logits.
        """
        y = y.clip(1e-3, 1 - 1e-3)

        return np.log(y / (1 - y))

    def _fit(self, X, y, sample_weight=None):
        """Build a Linear Boosting from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, ) or also (n_samples, n_targets) for
            multitarget regression.
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples, ), default=None
            Sample weights.

        Returns
        -------
        self : object
        """
        if not hasattr(self.base_estimator, 'fit_intercept'):
            raise ValueError("Only linear models are accepted as base_estimator. "
                             "Select one from linear_model class of scikit-learn.")

        if not is_regressor(self.base_estimator):
            raise ValueError("Select a regressor linear model as base_estimator.")

        n_sample, self.n_features_in_ = X.shape

        if hasattr(self, 'classes_'):
            class_to_int = dict(map(reversed, enumerate(self.classes_)))
            y = np.array([class_to_int[i] for i in y])
            y = self._inv_sigmoid(y)

        self.base_estimator_ = deepcopy(self.base_estimator)
        self.base_estimator_.fit(X, y, sample_weight)
        resid = y - self.base_estimator_.predict(X)

        criterion = 'squared_error' if _sklearn_v1 else 'mse'

        self.forest_estimator_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples
        )
        self.forest_estimator_.fit(X, resid, sample_weight)

        if hasattr(self.base_estimator_, 'coef_'):
            self.coef_ = self.base_estimator_.coef_

        if hasattr(self.base_estimator_, 'intercept_'):
            self.intercept_ = self.base_estimator_.intercept_

        self.feature_importances_ = self.forest_estimator_.feature_importances_

        return self

    def apply(self, X):
        """Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        check_is_fitted(self, attributes='base_estimator_')

        return self.forest_estimator_.apply(X)

    def decision_path(self, X):
        """Return the decision path in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.

        n_nodes_ptr : ndarray of shape (n_estimators + 1, )
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        check_is_fitted(self, attributes='base_estimator_')

        return self.forest_estimator_.decision_path(X)