import numbers
import numpy as np
import scipy.sparse as sp

from copy import deepcopy
from itertools import product
from joblib import Parallel, effective_n_jobs #, delayed
 
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from sklearn.utils import check_array
from sklearn.utils.validation import has_fit_parameter, check_is_fitted, \
    _check_sample_weight

from ._criterion import mse, rmse, mae, poisson
from ._criterion import hamming, crossentropy


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


def _partition_columns(n_columns, n_jobs):
    """Private function to partition columns splitting between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_columns)

    # Partition columns between jobs
    n_columns_per_job = np.full(n_jobs, n_columns // n_jobs, dtype=int)
    n_columns_per_job[:n_columns % n_jobs] += 1
    columns_per_job = np.cumsum(n_columns_per_job)
    columns_per_job = np.split(np.arange(n_columns), columns_per_job)
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
    largs = {}
    
    if n_sample < _self._min_samples_split:
        return loss, split_t, split_col, left_node, right_node

    for col,_bin in zip(split_feat, bins): 
        
        for q in _bin:
            
            # create 1D bool mask for right/left children
            mask = (X[:,col] > q)

            n_left, n_right = (~mask).sum(), mask.sum()

            if n_left < _self._min_samples_leaf or n_right < _self._min_samples_leaf:
                continue
            
            # create 2D bool mask for right/left children
            left_mesh = np.ix_(~mask, _self.features_id_)
            right_mesh = np.ix_(mask, _self.features_id_)
            
            # initilize model for left child
            classes_left_ = None
            model_left = deepcopy(_self.base_estimator)
            if hasattr(_self, 'classes_'):
                classes_left_ = np.unique(y[~mask])
                if len(classes_left_) == 1:
                    model_left = DummyClassifier(strategy="most_frequent")
                largs = {'classes': classes_left_}
            
            # fit model for left child
            if weights is None:
                model_left.fit(X[left_mesh], y[~mask])
                loss_left = feval(model_left, X[left_mesh], y[~mask], **largs) 
                _loss_left = loss_left * (n_left / n_sample)
            else:
                if support_sample_weight:
                    model_left.fit(X[left_mesh], y[~mask], 
                                   sample_weight=weights[~mask])
                else:
                    model_left.fit(X[left_mesh], y[~mask])
                loss_left = feval(model_left, X[left_mesh], y[~mask], 
                                  weights=weights[~mask], **largs) 
                _loss_left = loss_left * (weights[~mask].sum() / weights.sum())
            
            # initilize model for right child
            classes_right_ = None
            model_right = deepcopy(_self.base_estimator)
            if hasattr(_self, 'classes_'):
                classes_right_ = np.unique(y[mask])
                if len(classes_right_) == 1:
                    model_right = DummyClassifier(strategy="most_frequent")
                largs = {'classes': classes_right_}
            
            # fit model for right child
            if weights is None:
                model_right.fit(X[right_mesh], y[mask])
                loss_right = feval(model_right, X[right_mesh], y[mask], **largs) 
                _loss_right = loss_right * (n_right / n_sample)
            else:
                if support_sample_weight:
                    model_right.fit(X[right_mesh], y[mask], 
                                    sample_weight=weights[mask])
                else:
                    model_right.fit(X[right_mesh], y[mask])
                loss_right = feval(model_right, X[right_mesh], y[mask], 
                                   weights=weights[mask], **largs) 
                _loss_right = loss_right * (weights[mask].sum() / weights.sum())

            total_loss = _loss_left + _loss_right
            
            # store if best
            if total_loss < loss:
                split_t = q
                split_col = col
                loss = total_loss
                left_node = (deepcopy(model_left), loss_left, 
                             n_left, classes_left_)
                right_node = (deepcopy(model_right), loss_right, 
                              n_right, classes_right_)
    
    return loss, split_t, split_col, left_node, right_node


def _map_node(X, feat, direction, split):
    """Utility to map samples to nodes"""
    if direction == 'L':
        mask = (X[:,feat] <= split)
    else:
        mask = (X[:,feat] > split)
            
    return mask


def _predict_branch(X, branch_history):
    """Utility to map samples to branches"""
    mask = np.repeat(True, X.shape[0])

    for node in branch_history:
        mask = np.logical_and(_map_node(X, *node), mask)
    
    return mask


def _to_skip(branch, pruned_branches):
    """Skip pruned branches before explore it"""
    for p in pruned_branches:
        
        if branch[:len(p)] == p:
            return True
        
    return False


class _LinearTree(BaseEstimator):
    """Base class for Linear Tree meta-estimator.
    
    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self,
                 base_estimator,
                 criterion,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 max_bins,
                 categorical_features,
                 n_jobs): 
        
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.categorical_features = categorical_features
        self.n_jobs = n_jobs
        
    def _parallel_args(self):
        return {}
        
    def _split(self, X, y, bins,
               support_sample_weight,
               weights = None, 
               loss = None): 
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
        feval = CRITERIA[self.criterion]
        largs = {}
                
        # initialize first fit
        if loss is None:
            model = deepcopy(self.base_estimator)
            if weights is None or not support_sample_weight:
                model.fit(X[:,self.features_id_], y)
            else:
                model.fit(X[:,self.features_id_], y, sample_weight=weights)
                
            if hasattr(self, 'classes_'):
                largs['classes'] = self.classes_
                self._classes[''] = self.classes_
                
            loss = feval(model, X[:,self.features_id_], y, 
                         weights=weights, **largs)
            self._n_samples[''] = X.shape[0]
            self._models[''] = model
            self._losses[''] = loss
        
        # Parallel loops
        n_jobs, split_feat = _partition_columns(self.n_features_in_, self.n_jobs)
        
        # partition columns splittings between jobs
        all_results = Parallel(n_jobs=n_jobs, verbose=0,
                               **self._parallel_args())(
            delayed(_parallel_binning_fit)(
                feat,
                self, X, y,
                weights, support_sample_weight,
                bins[feat[0]:(feat[-1]+1)], 
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
            left_node = (None, None, None, None)
            right_node = (None, None, None, None) 

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
        feat = np.arange(self.n_features_in_)
        self.features_id_ = feat[~np.isin(feat, self._categorical_features)]
        
        # extract quantiles
        bins = np.linspace(0,1, self.max_bins)[1:-1]
        bins = np.quantile(X, bins, axis=0, interpolation='midpoint')
        bins = list(bins.T)
        bins = [np.unique(q) if c in self.features_id_ 
                else np.unique(X[:,c]) for c,q in enumerate(bins)]
        
        # check if base_estimator supports fitting with sample_weights
        support_sample_weight = has_fit_parameter(self.base_estimator,
                                                  "sample_weight")
                
        # simulate all the branch paths until the leaves
        branches = product(['L','R'], repeat=self.max_depth)
        # initialize empty list of pruned branches
        pruned = []
        # store the results of each node in dicts
        self._thresholds = {}
        self._n_samples = {}
        self._history = {}
        self._losses = {}
        self._models = {}
        
        # in the beginning consider all the samples
        start = np.repeat(True, n_sample)
        
        for b in branches:

            path = ''
            loss = None
            branch = ''.join(b)
            
            # skip pruned branches
            if _to_skip(branch, pruned):
                continue
            
            self._history[branch] = []
            mask = start.copy()

            for n in branch: 
                
                inv_n = 'L' if n == 'R' else 'R'
                inv_path = path+inv_n
                path += n
                
                # node previusly explored
                if path in self._losses:
                    self._history[branch].append(self._thresholds[path])
                    mask = np.logical_and(
                        mask, _map_node(X, *self._thresholds[path]))
                    loss = self._losses[path]
                
                # node explored for the first time
                else:
                    # find the best split
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
                    if split_col is None:
                        self._losses[path] = None
                        self._losses[inv_path] = None
                        pruned.extend([path, inv_path])
                        break
                    
                    # store results
                    model_left, loss_left, n_left, class_left = left_node
                    model_right, loss_right, n_right, class_right = right_node
                    mask = np.logical_and(
                        mask, _map_node(X, split_col, n, split_t))

                    if n == 'L':
                        loss = loss_left
                        self._models[path] = deepcopy(model_left)
                        self._models[inv_path] = deepcopy(model_right)
                        self._losses[path] = loss_left
                        self._losses[inv_path] = loss_right
                        self._n_samples[path] = n_left
                        self._n_samples[inv_path] = n_right
                        if hasattr(self, 'classes_'):
                            self._classes[path] = class_left
                            self._classes[inv_path] = class_right
                    else:
                        loss = loss_right
                        self._models[path] = deepcopy(model_right)
                        self._models[inv_path] = deepcopy(model_left)
                        self._losses[path] = loss_right
                        self._losses[inv_path] = loss_left
                        self._n_samples[path] = n_right
                        self._n_samples[inv_path] = n_left
                        if hasattr(self, 'classes_'):
                            self._classes[path] = class_right
                            self._classes[inv_path] = class_left

                    self._history[branch].append((split_col, n, split_t))
                    self._thresholds[path] = (split_col, n, split_t)
                    self._thresholds[inv_path] = (split_col, inv_n, split_t)

            # when the branch stop growing
            if self._losses[path] is None:
                self._history[path[:-1]] = self._history[branch]
                del self._history[branch]
                
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
        if isinstance(self.min_samples_split, numbers.Integral): 
            if self.min_samples_split < 6:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 5 or a float in (0.0, 1.0); "
                                 "got the integer {}".format(self.min_samples_split))
            self._min_samples_split = self.min_samples_split
        else: 
            if not 0. < self.min_samples_split < 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 5 or a float in (0.0, 1.0); "
                                 "got the float {}".format(self.min_samples_split))

            self._min_samples_split = int(np.ceil(self.min_samples_split * X.shape[0]))
            self._min_samples_split = max(6, self._min_samples_split)
        
        if isinstance(self.min_samples_leaf, numbers.Integral): 
            if self.min_samples_leaf < 3:
                raise ValueError("min_samples_leaf must be an integer "
                                 "greater than 2 or a float in (0.0, 1.0); "
                                 "got the integer {}".format(self.min_samples_leaf))
            self._min_samples_leaf = self.min_samples_leaf
        else: 
            if not 0. < self.min_samples_leaf < 1.:
                raise ValueError("min_samples_leaf must be an integer "
                                 "greater than 2 or a float in (0.0, 1.0); "
                                 "got the float {}".format(self.min_samples_leaf))

            self._min_samples_leaf = int(np.ceil(self.min_samples_leaf * X.shape[0]))
            self._min_samples_leaf = max(3, self._min_samples_leaf)
            
        if not 1 <= self.max_depth <= 20:
            raise ValueError("max_depth must be an integer in [1, 20].")
            
        if not 10 <= self.max_bins <= 120:
            raise ValueError("max_bins must be an integer in [10, 120].")
        
        if not hasattr(self.base_estimator, 'fit_intercept'):
            raise ValueError("Only linear models are accepted as base_estimator. "
                             "Select one from linear_model class of scikit-learn.")
            
        if self.categorical_features is not None:
            categorical_features = np.unique(self.categorical_features)
            
            if not issubclass(categorical_features.dtype.type, numbers.Integral):
                raise ValueError(
                    "No valid specification of categorical columns. "
                    "Only a scalar, list or array-like of all "
                    "integers is allowed."
                )

            if ((categorical_features < 0).any() or 
                (categorical_features >= X.shape[1]).any()):
                raise ValueError(
                    'all categorical features must be in [0, {}].'.format(
                        X.shape[1] - 1)
                )
                
            if len(categorical_features) == X.shape[1]:
                raise ValueError(
                    "Only categorical features detected. "
                    "No features available for fitting."
                )
        else:
            categorical_features = [] 
        self._categorical_features = categorical_features
            
        self._grow(X, y, sample_weight)
        
        return self
    
    def _get_node_ids(self):
        """Map each node uniquely with an integer. 
        
        Returns
        -------
        split_nodes : dict, split nodes mapping
        leaf_nodes : dict, leaf nodes mapping
        """
        if len(self._thresholds) > 0:
            split_nodes = set(map(lambda x: x[:-1], self._thresholds.keys()))
            leaf_nodes = set(self._history.keys())
            
            nodes = sorted(split_nodes.union(leaf_nodes))
            nodes = dict(map(reversed, enumerate(nodes)))
            
            split_nodes = {n:nodes[n] for n in sorted(split_nodes)}
            leaf_nodes = {n:nodes[n] for n in sorted(leaf_nodes)}
            
        else:
            split_nodes = None
            leaf_nodes = {'': 0}
        
        return split_nodes, leaf_nodes
    
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
        nodes : nested dict
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

        check_is_fitted(self, attributes='_models')
        
        if max_depth is None:
            max_depth = 20
        if max_depth < 1:
            raise ValueError("max_depth bust be > 0, "
                             "given {}".format(max_depth))
        
        split_nodes, leaf_nodes = self._get_node_ids()
        nodes = {}
        
        if split_nodes is not None and not only_leaves:
            
            if (feature_names is not None and
                len(feature_names) != self.n_features_in_):
                raise ValueError("feature_names must contain "
                                 "{} elements, got {}".format(
                                     self.n_features_in_, len(feature_names)))

            if feature_names is None:
                feature_names = np.arange(self.n_features_in_)
            
            for n in split_nodes:
                
                if len(n) >= max_depth:
                    continue
                
                n_l, n_r = n+'L', n+'R'                
                child_l = (split_nodes[n_l] if n_l in split_nodes 
                           else leaf_nodes[n_l])
                child_r = (split_nodes[n_r] if n_r in split_nodes 
                           else leaf_nodes[n_r])
    
                nodes[split_nodes[n]] = {
                    'col': feature_names[self._thresholds[n_l][0]],
                    'th': round(self._thresholds[n_l][-1], 4),
                    'loss': round(self._losses[n_l] + self._losses[n_r], 4),
                    'samples': self._n_samples[n_l] + self._n_samples[n_r],
                    'children': (child_l, child_r),
                    'models': (self._models[n_l], self._models[n_r])
                }
        
        for n in leaf_nodes:
            
            if len(n) > max_depth:
                continue
            
            nodes[leaf_nodes[n]] = {
                'loss': round(self._losses[n], 4),
                'samples': self._n_samples[n],
                'models': self._models[n]
            } 
            
            if hasattr(self, 'classes_'):
                nodes[leaf_nodes[n]]['classes'] = self._classes[n]

        return nodes
    
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
        check_is_fitted(self, attributes='_models')
        X = check_array(
            X, accept_sparse=False, dtype=None,
            force_all_finite=False)
        self._check_n_features(X, reset=False)
        
        split_nodes, leaf_nodes = self._get_node_ids()
        X_leaves = np.zeros(X.shape[0], dtype='int64')

        for b in self._history:

            mask = _predict_branch(X, self._history[b])
            if (~mask).all():
                continue

            X_leaves[mask] = leaf_nodes[b]
            
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
        check_is_fitted(self, attributes='_models')
        X = check_array(
            X, accept_sparse=False, dtype=None,
            force_all_finite=False)
        self._check_n_features(X, reset=False)
        
        split_nodes, leaf_nodes = self._get_node_ids()
        all_nodes = leaf_nodes.copy()
        if split_nodes is not None:
            all_nodes.update(split_nodes)
        indicator = np.zeros((X.shape[0], len(all_nodes)), dtype='int64')

        for b in self._history:

            mask = _predict_branch(X, self._history[b])
            if (~mask).all():
                continue

            path = ''
            paths_id = []
            for n in b:
                paths_id.append(all_nodes[path])
                path += n
            paths_id.append(all_nodes[path])

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
    
    
class LinearTreeRegressor(_LinearTree, RegressorMixin):
    """A Linear Tree Regressor.
    
    A Linear Tree Regressor is a meta-estimator that combine the learning
    ability of Decision Tree and the predictive power of Linear Models.
    Like in tree-based algorithms, the received data are split according to  
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
        The function to measure the quality of a split. `"poisson"`
        requires `y >= 0`.
        
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
          
    n_jobs : int, default=None
        The number of jobs to run in parallel for model fitting. 
        ``None`` means 1 using one processor. ``-1`` means using all
        processors. 
        
    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed.
        
    features_id_ : ndarray of int
        The integer ids of features when :meth:`fit` is performed.
        This not include categorical_features.
        
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
    def __init__(self,
                 base_estimator,
                 criterion = 'mse',
                 max_depth = 5,
                 min_samples_split = 6,
                 min_samples_leaf = 0.1,
                 max_bins = 25,
                 categorical_features = None,
                 n_jobs = None): 
        
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.categorical_features = categorical_features
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
            raise ValueError("Regression tasks supports only criterion in {}, "
                             "got '{}'.".format(reg_criterions, self.criterion))
            
        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y, accept_sparse=False, dtype=None,
            force_all_finite=False, multi_output=True
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)
        
        y_shape = np.shape(y)
        self.n_targets_ = y_shape[1] if len(y_shape) > 1 else 1
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
        check_is_fitted(self, attributes='_models')
        X = check_array(
            X, accept_sparse=False, dtype=None,
            force_all_finite=False)
        self._check_n_features(X, reset=False)
        
        if self.n_targets_ > 1:
            pred = np.zeros((X.shape[0], self.n_targets_))
        else:
            pred = np.zeros(X.shape[0])

        for b in self._history:

            mask = _predict_branch(X, self._history[b])
            if (~mask).all():
                continue
            
            pred[mask] = self._models[b].predict(X[np.ix_(mask, self.features_id_)])
            
        return pred
    
    
class LinearTreeClassifier(_LinearTree, ClassifierMixin):
    """A Linear Tree Classifier.
    
    A Linear Tree Classifier is a meta-estimator that combine the learning
    ability of Decision Tree and the predictive power of Linear Models.
    Like in tree-based algorithms, the received data are split according to 
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
          
    n_jobs : int, default=None
        The number of jobs to run in parallel for model fitting. 
        ``None`` means 1 using one processor. ``-1`` means using all
        processors. 
        
    Attributes
    ----------
    n_features_in_ : int
        The number of features when :meth:`fit` is performed. 
        
    features_id_ : ndarray of int
        The integer ids of features when :meth:`fit` is performed. 
        This not include categorical_features. 
        
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
    def __init__(self,
                 base_estimator,
                 criterion = 'hamming',
                 max_depth = 5,
                 min_samples_split = 6,
                 min_samples_leaf = 0.1,
                 max_bins = 25,
                 categorical_features = None,
                 n_jobs = None): 
        
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.categorical_features = categorical_features
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
            raise ValueError("Classification tasks supports only criterion in {}, "
                             "got '{}'.".format(clas_criterions, self.criterion))
            
        if (not hasattr(self.base_estimator, 'predict_proba') and
           self.criterion == 'crossentropy'):
            raise ValueError("The 'crossentropy' criterion requires a base_estimator "
                             "with predict_proba method.")

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y, accept_sparse=False, dtype=None,
            force_all_finite=False, multi_output=False
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)
        
        self._classes = {}
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
        check_is_fitted(self, attributes='_models')
        X = check_array(
            X, accept_sparse=False, dtype=None,
            force_all_finite=False)
        self._check_n_features(X, reset=False)
            
        pred = np.empty(X.shape[0], dtype=self.classes_.dtype)

        for b in self._history:

            mask = _predict_branch(X, self._history[b])
            if (~mask).all():
                continue
                
            pred[mask] = self._models[b].predict(X[np.ix_(mask, self.features_id_)])
            
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
        check_is_fitted(self, attributes='_models')
        X = check_array(
            X, accept_sparse=False, dtype=None,
            force_all_finite=False)
        self._check_n_features(X, reset=False)

        pred = np.zeros((X.shape[0], len(self.classes_)))
        
        if hasattr(self.base_estimator, 'predict_proba'):
            for b in self._history:

                mask = _predict_branch(X, self._history[b])
                if (~mask).all():
                    continue
                    
                pred[np.ix_(mask, np.isin(self.classes_, self._classes[b]))] = \
                    self._models[b].predict_proba(X[np.ix_(mask, self.features_id_)])
        
        else:
            pred_class = self.predict(X)
            class_to_int = dict(map(reversed, enumerate(self.classes_))) 
            pred_class = np.array([class_to_int[v] for v in pred_class])
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