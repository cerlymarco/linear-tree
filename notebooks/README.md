# API Reference

## LinearTreeRegressor
```
class lineartree.LinearTreeRegressor(base_estimator, criterion = 'mse', max_depth = 5, min_samples_split = 6, min_samples_leaf = 0.1, max_bins = 25, categorical_features = None, n_jobs = None)
```

#### Parameters:
- ```base_estimator : object```
    
    The base estimator to fit on dataset splits.
    The base estimator must be a sklearn.linear_model. 
    
- ```criterion : {"mse", "rmse", "mae", "poisson"}, default="mse"```
    
    The function to measure the quality of a split. `"poisson"` requires `y >= 0`.
    
- ```max_depth : int, default=5```

    The maximum depth of the tree considering only the splitting nodes.
    A higher value implies a higher training time.
    
- ```min_samples_split : int or float, default=6```

    The minimum number of samples required to split an internal node.
    The minimum valid number of samples in each node is 6.
    A lower value implies a higher training time.
    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.

- ```min_samples_leaf : int or float, default=0.1```

    The minimum number of samples required to be at a leaf node. 
    A split point at any depth will only be considered if it leaves at least `min_samples_leaf` training samples in each of the left and right branches.
    The minimum valid number of samples in each leaf is 3.
    A lower value implies a higher training time.
    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and
    `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.

- ```max_bins : int, default=25```

    The maximum number of bins to use to search the optimal split in each feature. Features with a small number of unique values may use less than ``max_bins`` bins. Must be lower than 120 and larger than 10. 
    A higher value implies a higher training time. 

- ```categorical_features : int or array-like of int, default=None```

    Indicates the categorical features. 
    All categorical indices must be in `[0, n_features)`. 
    Categorical features are used for splits but are not used in model fitting.
    More categorical features imply a higher training time.
    - None : no feature will be considered categorical.
    - integer array-like : integer indices indicating categorical features.
    - integer : integer index indicating a categorical feature.

- ```n_jobs : int, default=None```

    The number of jobs to run in parallel for model fitting. ``None`` means 1 using one processor. ``-1`` means using all processors. 

#### Attributes:
- ```n_features_in_ : int```

    The number of features when :meth:`fit` is performed.
    
- ```features_id_ : ndarray of int```

    The integer ids of features when :meth:`fit` is performed.
    This not include categorical_features.
    
- ```n_targets_ : int```

    The number of targets when :meth:`fit` is performed.

#### Methods:
- ```fit(X, y, sample_weight=None)```

    Build a Linear Tree of a linear estimator from the training set (X, y).
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        The training input samples.  
    - `y` : array-like of shape (n_samples, ) or (n_samples, n_targets)
        Target values.
    - `sample_weight` : array-like of shape (n_samples, ), default=None
        Sample weights. If None, then samples are equally weighted. Note that if the base estimator does not support sample weighting, the sample weights are still used to evaluate the splits.
    
    **Returns:**
    
    - `self` : object

- ```predict(X)```

    Predict regression target for X.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        Samples. 
    
    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, ) or also (n_samples, n_targets) if multitarget regression 
        The predicted values.

- ```apply(X)```

    Return the index of the leaf that each sample is predicted as.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        Samples. 
    
    **Returns:**
    
    - `X_leaves` : array-like of shape (n_samples, ) 
        For each datapoint x in X, return the index of the leaf x ends up in. Leaves are numbered within ``[0; n_nodes)``, possibly with gaps in the numbering.

- ```decision_path(X)```

    Return the decision path in the tree.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        Samples. 
    
    **Returns:**
    
    - `indicator` : sparse matrix of shape (n_samples, n_nodes) 
        Return a node indicator CSR matrix where non zero elements indicates that the samples goes through the nodes.

- ```summary(feature_names=None, only_leaves=False, max_depth=None)```

    Return a summary of nodes created from model fitting.
    
    **Parameters:**
    
    - `feature_names` : array-like of shape (n_features, ), default=None
        Names of each of the features. If None, generic names will be used (“X[0]”, “X[1]”, …). 
    - `only_leaves` : bool, default=False
        Store only information of leaf nodes.
    - `max_depth` : int, default=None
        The maximum depth of the representation. If None, the tree is fully generated.
    
    **Returns:**
    
    - `nodes` : nested dict  
        The keys are the integer map of each node. 
        The values are dicts containing information for that node:
        - 'col' (^): column used for splitting;
        - 'th' (^): threshold value used for splitting in the selected column;
        - 'loss': loss computed at node level. Weighted sum of children' losses if it is a splitting node;
        - 'samples': number of samples in the node. Sum of children' samples if it is a split node;
        - 'children' (^): integer mapping of possible children nodes;
        - 'models': fitted linear models built in each split. Single model if it is leaf node;
        - 'classes' (^^): target classes detected in the split. Available only for LinearTreeClassifier.

        (^): Only for split nodes. 
        (^^): Only for leaf nodes.

- ```model_to_dot(feature_names=None, max_depth=None)```

    Convert a fitted Linear Tree model to dot format. 
    It results in ModuleNotFoundError if graphviz or pydot are not available.
    When installing graphviz make sure to add it to the system path.
    
    **Parameters:**
    
    - `feature_names` : array-like of shape (n_features, ), default=None
        Names of each of the features. If None, generic names will be used (“X[0]”, “X[1]”, …). 
    - `max_depth` : int, default=None
        The maximum depth of the representation. If None, the tree is fully generated.
    
    **Returns:**
    
    - `graph` : pydot.Dot instance  
        Return an instance representing the Linear Tree. Splitting nodes have a rectangular shape while leaf nodes have a circular one.

- ```plot_model(feature_names=None, max_depth=None)```

    Convert a fitted Linear Tree model to dot format and display it. 
    It results in ModuleNotFoundError if graphviz or pydot are not available.
    When installing graphviz make sure to add it to the system path.
    
    **Parameters:**
    
    - `feature_names` : array-like of shape (n_features, ), default=None
        Names of each of the features. If None, generic names will be used (“X[0]”, “X[1]”, …). 
    - `max_depth` : int, default=None
        The maximum depth of the representation. If None, the tree is fully generated.
    
    **Returns:**
    
    - A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks. Splitting nodes have a rectangular shape while leaf nodes have a circular one.


## LinearTreeClassifier
```
class lineartree.LinearTreeClassifier(base_estimator, criterion = 'hamming', max_depth = 5, min_samples_split = 6,  min_samples_leaf = 0.1, max_bins = 25, categorical_features = None, n_jobs = None)
```

#### Parameters:

- ```base_estimator : object```

    The base estimator to fit on dataset splits.
    The base estimator must be a sklearn.linear_model. 
    The selected base estimator is automatically substituted by a `~sklearn.dummy.DummyClassifier` when a dataset split is composed of unique labels.

- ```criterion : {"hamming", "crossentropy"}, default="hamming"```

    The function to measure the quality of a split. `"crossentropy"` can be used only if `base_estimator` has `predict_proba` method

- ```max_depth : int, default=5```

    The maximum depth of the tree considering only the splitting nodes.
    A higher value implies a higher training time.
    
- ```min_samples_split : int or float, default=6```

    The minimum number of samples required to split an internal node.
    The minimum valid number of samples in each node is 6.
    A lower value implies a higher training time.
    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.

- ```min_samples_leaf : int or float, default=0.1```

    The minimum number of samples required to be at a leaf node. 
    A split point at any depth will only be considered if it leaves at least `min_samples_leaf` training samples in each of the left and right branches.
    The minimum valid number of samples in each leaf is 3.
    A lower value implies a higher training time.
    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and
    `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.

- ```max_bins : int, default=25```

    The maximum number of bins to use to search the optimal split in each feature. Features with a small number of unique values may use less than ``max_bins`` bins. Must be lower than 120 and larger than 10. 
    A higher value implies a higher training time. 

- ```categorical_features : int or array-like of int, default=None```

    Indicates the categorical features. 
    All categorical indices must be in `[0, n_features)`. 
    Categorical features are used for splits but are not used in model fitting.
    More categorical features imply a higher training time.
    - None : no feature will be considered categorical.
    - integer array-like : integer indices indicating categorical features.
    - integer : integer index indicating a categorical feature.
    
- ```n_jobs : int, default=None```

    The number of jobs to run in parallel for model fitting. ``None`` means 1 using one processor. ``-1`` means using all processors. 

#### Attributes:

- ```n_features_in_ : int```

    The number of features when :meth:`fit` is performed.
    
- ```features_id_ : ndarray of int```

    The integer ids of features when :meth:`fit` is performed.
    This not include categorical_features.
    
- ```classes_ : ndarray of shape (n_classes, )```

    A list of class labels known to the classifier.

#### Methods:

- ```fit(X, y, sample_weight=None)```

    Build a Linear Tree of a linear estimator from the training set (X, y).
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        The training input samples.  
    - `y` : array-like of shape (n_samples, ) or (n_samples, n_targets)
        Target values.
    - `sample_weight` : array-like of shape (n_samples, ), default=None
        Sample weights. If None, then samples are equally weighted. Note that if the base estimator does not support sample weighting, the sample weights are still used to evaluate the splits.
    
    **Returns:**
    
    - `self` : object

- ```predict(X)```

    Predict class for X.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        Samples. 
    
    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, ) 
        The predicted classes.

- ```predict_proba(X)```

    Predict class probabilities for X.
    If base estimators do not implement a ``predict_proba`` method, then the one-hot encoding of the predicted class is returned
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        Samples. 
    
    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, n_classes) 
        The class probabilities of the input samples. The order of the classes corresponds to that in the attribute :term:`classes_`.

- ```predict_log_proba(X)```

    Predict class log-probabilities for X.
    If base estimators do not implement a ``predict_log_proba`` method, then the logarithm of the one-hot encoded predicted class is returned.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        Samples. 
    
    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, n_classes) 
        The class log-probabilities of the input samples. The order of the classes corresponds to that in the attribute :term:`classes_`.

- ```apply(X)```

    Return the index of the leaf that each sample is predicted as.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        Samples. 
    
    **Returns:**
    
    - `X_leaves` : array-like of shape (n_samples, ) 
        For each datapoint x in X, return the index of the leaf x ends up in. Leaves are numbered within ``[0; n_nodes)``, possibly with gaps in the numbering.

- ```decision_path(X)```

    Return the decision path in the tree.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        Samples. 
    
    **Returns:**
    
    - `indicator` : sparse matrix of shape (n_samples, n_nodes) 
        Return a node indicator CSR matrix where non zero elements indicates that the samples goes through the nodes.

- ```summary(feature_names=None, only_leaves=False, max_depth=None)```

    Return a summary of nodes created from model fitting.
    
    **Parameters:**
    
    - `feature_names` : array-like of shape (n_features, ), default=None
        Names of each of the features. If None, generic names will be used (“X[0]”, “X[1]”, …). 
    - `only_leaves` : bool, default=False
        Store only information of leaf nodes.
    - `max_depth` : int, default=None
        The maximum depth of the representation. If None, the tree is fully generated.
    
    **Returns:**
    
    - `nodes` : nested dict  
        The keys are the integer map of each node. 
        The values are dicts containing information for that node:
        - 'col' (^): column used for splitting;
        - 'th' (^): threshold value used for splitting in the selected column;
        - 'loss': loss computed at node level. Weighted sum of children' losses if it is a splitting node;
        - 'samples': number of samples in the node. Sum of children' samples if it is a split node;
        - 'children' (^): integer mapping of possible children nodes;
        - 'models': fitted linear models built in each split. Single model if it is leaf node;
        - 'classes' (^^): target classes detected in the split. Available only for LinearTreeClassifier.

        (^): Only for split nodes. 
        (^^): Only for leaf nodes.

- ```model_to_dot(feature_names=None, max_depth=None)```

    Convert a fitted Linear Tree model to dot format. 
    It results in ModuleNotFoundError if graphviz or pydot are not available.
    When installing graphviz make sure to add it to the system path.
    
    **Parameters:**
    
    - `feature_names` : array-like of shape (n_features, ), default=None
        Names of each of the features. If None, generic names will be used (“X[0]”, “X[1]”, …). 
    - `max_depth` : int, default=None
        The maximum depth of the representation. If None, the tree is fully generated.
    
    **Returns:**
    
    - `graph` : pydot.Dot instance  
        Return an instance representing the Linear Tree. Splitting nodes have a rectangular shape while leaf nodes have a circular one.

- ```plot_model(feature_names=None, max_depth=None)```

    Convert a fitted Linear Tree model to dot format and display it. 
    It results in ModuleNotFoundError if graphviz or pydot are not available.
    When installing graphviz make sure to add it to the system path.
    
    **Parameters:**
    
    - `feature_names` : array-like of shape (n_features, ), default=None
        Names of each of the features. If None, generic names will be used (“X[0]”, “X[1]”, …). 
    - `max_depth` : int, default=None
        The maximum depth of the representation. If None, the tree is fully generated.
    
    **Returns:**
    
    - A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks. Splitting nodes have a rectangular shape while leaf nodes have a circular one.