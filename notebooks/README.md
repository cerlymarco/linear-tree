# API Reference

## LinearTreeRegressor
```
class lineartree.LinearTreeRegressor(base_estimator, *, criterion = 'mse', max_depth = 5, min_samples_split = 6, min_samples_leaf = 0.1, max_bins = 25, categorical_features = None, split_features = None, linear_features = None, n_jobs = None)
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
    
- ```split_features : int or array-like of int, default=None```
        
    Defines which features can be used to split on.
    All split feature indices must be in `[0, n_features)`.
    - None : All features will be used for splitting.
    - integer array-like : integer indices indicating splitting features.
    - integer : integer index indicating a single splitting feature.
    
- ```linear_features : int or array-like of int, default=None```

    Defines which features are used for the linear model in the leaves.
    All linear feature indices must be in `[0, n_features)`.
    - None : All features except those in `categorical_features` will be used in the leaf models.
    - integer array-like : integer indices indicating features to be used in the leaf models.
    - integer : integer index indicating a single feature to be used in the leaf models.

- ```n_jobs : int, default=None```

    The number of jobs to run in parallel for model fitting. ``None`` means 1 using one processor. ``-1`` means using all processors. 

#### Attributes:

- ```n_features_in_ : int```

    The number of features when :meth:`fit` is performed.
    
- ```feature_importances_ : ndarray of shape (n_features, )```

    Normalized total reduction of criteria by splitting features.
    
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
    
    - `summary` : nested dict 
     
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
class lineartree.LinearTreeClassifier(base_estimator, *, criterion = 'hamming', max_depth = 5, min_samples_split = 6,  min_samples_leaf = 0.1, max_bins = 25, categorical_features = None, split_features = None, linear_features = None, n_jobs = None)
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
    
- ```split_features : int or array-like of int, default=None```
        
    Defines which features can be used to split on.
    All split feature indices must be in `[0, n_features)`.
    - None : All features will be used for splitting.
    - integer array-like : integer indices indicating splitting features.
    - integer : integer index indicating a single splitting feature.
    
- ```linear_features : int or array-like of int, default=None```

    Defines which features are used for the linear model in the leaves.
    All linear feature indices must be in `[0, n_features)`.
    - None : All features except those in `categorical_features` will be used in the leaf models.
    - integer array-like : integer indices indicating features to be used in the leaf models.
    - integer : integer index indicating a single feature to be used in the leaf models.
    
- ```n_jobs : int, default=None```

    The number of jobs to run in parallel for model fitting. ``None`` means 1 using one processor. ``-1`` means using all processors. 

#### Attributes:

- ```n_features_in_ : int```

    The number of features when :meth:`fit` is performed.
    
- ```feature_importances_ : ndarray of shape (n_features, )```

    Normalized total reduction of criteria by splitting features.
    
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
    
    - `summary` : nested dict
      
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
        
        
## LinearBoostRegressor
```
class lineartree.LinearBoostRegressor(base_estimator, *, loss = 'linear', n_estimators = 10, max_depth = 3, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = None, random_state = None, max_leaf_nodes = None, min_impurity_decrease = 0.0, ccp_alpha = 0.0)
```

#### Parameters:

- ```base_estimator : object```

    The base estimator iteratively fitted.
    The base estimator must be a sklearn.linear_model.
    
- ```loss : {"linear", "square", "absolute", "exponential"}, default="linear"```

    The function used to calculate the residuals of each sample.
    
- ```n_estimators : int, default=10```

    The number of boosting stages to perform. It corresponds to the number of the new features generated.
    
- ```max_depth : int, default=3```

    The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    
- ```min_samples_split : int or float, default=2```

    The minimum number of samples required to split an internal node:
    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.
    
- ```min_samples_leaf : int or float, default=1```
  
    The minimum number of samples required to be at a leaf node. 
    A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.
  
- ```min_weight_fraction_leaf : float, default=0.0```

    The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided. 
  
- ```max_features : int, float or {"auto", "sqrt", "log2"}, default=None```
  
    The number of features to consider when looking for the best split:
    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split.
    - If "auto", then `max_features=n_features`.
    - If "sqrt", then `max_features=sqrt(n_features)`.
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.
    Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than ``max_features`` features.  
  
- ```max_leaf_nodes : int, default=None```
  
    Grow a tree with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.
  
- ```min_impurity_decrease : float, default=0.0```
  
    A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
  
- ```ccp_alpha : non-negative float, default=0.0```

    Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ``ccp_alpha`` will be chosen. By default, no pruning is performed. See :ref:`minimal_cost_complexity_pruning` for details.  
  
#### Attributes:

- ```n_features_in_ : int```

    The number of features when :meth:`fit` is performed.
    
- ```n_features_out_ : int```

    The total number of features used to fit the base estimator in the last iteration. The number of output features is equal to the sum of n_features_in_ and n_estimators.
    
- ```coef_ : array of shape (n_features_out_, ) or (n_targets, n_features_out_)```

    Estimated coefficients for the linear regression problem.
    If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features_out_), while if only one target is passed, this is a 1D array of length n_features.

- ```intercept_ : float or array of shape (n_targets, )```

    Independent term in the linear model. Set to 0 if `fit_intercept = False` in `base_estimator`
    
#### Methods:

- ```fit(X, y, sample_weight=None)```

    Build a Linear Boosting from the training set (X, y).
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        The training input samples.  
    
    - `y` : array-like of shape (n_samples, ) or (n_samples, n_targets)
        
        Target values.
    
    - `sample_weight` : array-like of shape (n_samples, ), default=None
        
        Sample weights. 
        
    **Returns:**
    
    - `self` : object

- ```predict(X)```

    Predict regression target for X.

    **Parameters:**

    - `X` : array-like of shape (n_samples, n_features)
        
        Samples. 

    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, ) or also (n_samples, n_targets) if multitarget regression.
        
        The predicted values.
        
- ```transform(X)```

    Transform dataset.

    **Parameters:**

    - `X` : array-like of shape (n_samples, n_features)
        
        Input data to be transformed. Use ``dtype=np.float32`` for maximum efficiency. 

    **Returns:**
    
    - `X_transformed` : ndarray of shape (n_samples, n_out).
        
        Transformed dataset.
        `n_out` is equal to `n_features` + `n_estimators`.
        
## LinearBoostClassifier
```
class lineartree.LinearBoostClassifier(base_estimator, loss = 'hamming', n_estimators = 10, max_depth = 3, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = None, random_state = None, max_leaf_nodes = None, min_impurity_decrease = 0.0, ccp_alpha = 0.0)
```

#### Parameters:

- ```base_estimator : object```

    The base estimator iteratively fitted.
    The base estimator must be a sklearn.linear_model.
    
- ```loss : {"hamming", "entropy"}, default="hamming"```

    The function used to calculate the residuals of each sample.
    
- ```n_estimators : int, default=10```

    The number of boosting stages to perform. It corresponds to the number of the new features generated.
    
- ```max_depth : int, default=3```

    The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    
- ```min_samples_split : int or float, default=2```

    The minimum number of samples required to split an internal node:
    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.
    
- ```min_samples_leaf : int or float, default=1```
  
    The minimum number of samples required to be at a leaf node. 
    A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.
  
- ```min_weight_fraction_leaf : float, default=0.0```

    The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided. 
  
- ```max_features : int, float or {"auto", "sqrt", "log2"}, default=None```
  
    The number of features to consider when looking for the best split:
    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split.
    - If "auto", then `max_features=n_features`.
    - If "sqrt", then `max_features=sqrt(n_features)`.
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.
    
    Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than ``max_features`` features.  
  
- ```max_leaf_nodes : int, default=None```
  
    Grow a tree with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.
  
- ```min_impurity_decrease : float, default=0.0```
  
    A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

- ```ccp_alpha : non-negative float, default=0.0```

    Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ``ccp_alpha`` will be chosen. By default, no pruning is performed. See :ref:`minimal_cost_complexity_pruning` for details.  
  
#### Attributes:

- ```n_features_in_ : int```

    The number of features when :meth:`fit` is performed.
    
- ```n_features_out_ : int```

    The total number of features used to fit the base estimator in the last iteration. The number of output features is equal to the sum of n_features_in_ and n_estimators.
    
- ```coef_ : array of shape (n_features_out_, ) or (n_targets, n_features_out_)```

    Estimated coefficients for the linear regression problem.
    If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features_out_), while if only one target is passed, this is a 1D array of length n_features_out_.

- ```intercept_ : float or array of shape (n_targets, )```

    Independent term in the linear model. Set to 0 if `fit_intercept = False` in `base_estimator`
    
- ```classes_ : ndarray of shape (n_classes, )```

    A list of class labels known to the classifier.
    
#### Methods:

- ```fit(X, y, sample_weight=None)```

    Build a Linear Boosting from the training set (X, y).
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        The training input samples.  
    
    - `y` : array-like of shape (n_samples, ) or (n_samples, n_targets)
        
        Target values.
    
    - `sample_weight` : array-like of shape (n_samples, ), default=None
        
        Sample weights. 
        
    **Returns:**
    
    - `self` : object

- ```predict(X)```

    Predict class for X.

    **Parameters:**

    - `X` : array-like of shape (n_samples, n_features)
        
        Samples. 

    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, ) or also (n_samples, n_targets) if multitarget regression.
        
        The predicted classes.
        
- ```transform(X)```

    Transform dataset.

    **Parameters:**

    - `X` : array-like of shape (n_samples, n_features)
        
        Input data to be transformed. Use ``dtype=np.float32`` for maximum efficiency. 

    **Returns:**
    
    - `X_transformed` : ndarray of shape (n_samples, n_out)
        
        Transformed dataset.
        `n_out` is equal to `n_features` + `n_estimators`.
        
## LinearForestRegressor
```
class lineartree.LinearForestRegressor(base_estimator, *, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., bootstrap=True, oob_score=False, n_jobs=None, random_state=None, ccp_alpha=0.0, max_samples=None)
```

#### Parameters:

- ```base_estimator : object```

    The linear estimator fitted on the raw target.
    The linear estimator must be a regressor from sklearn.linear_model.
    
- ```n_estimators : int, default=100```

    The number of trees in the forest.
    
- ```max_depth : int, default=None```

    The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    
- ```min_samples_split : int or float, default=2```

    The minimum number of samples required to split an internal node:    
    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.
      
- ```min_samples_leaf : int or float, default=1```

    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.
      
- ```min_weight_fraction_leaf : float, default=0.0```

    The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
    
- ```max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"```

    The number of features to consider when looking for the best split:    
    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a fraction and `round(max_features * n_features)` features are considered at each split.
    - If "auto", then `max_features=n_features`.
    - If "sqrt", then `max_features=sqrt(n_features)`.
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`
    
    Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than ``max_features`` features.
    
- ```max_leaf_nodes : int, default=None```

    Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    
- ```min_impurity_decrease : float, default=0.0```

    A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

- ```bootstrap : bool, default=True```

    Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
    
- ```oob_score : bool, default=False```

    Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True.
    
- ```n_jobs : int, default=None```

    The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`, :meth:`decision_path` and :meth:`apply` are all parallelized over the trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
    
- ```random_state : int, RandomState instance or None, default=None```

    Controls both the randomness of the bootstrapping of the samples used when building trees (if ``bootstrap=True``) and the sampling of the features to consider when looking for the best split at each node (if ``max_features < n_features``).
    
- ```ccp_alpha : non-negative float, default=0.0```

    Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ``ccp_alpha`` will be chosen. By default, no pruning is performed. See :ref:`minimal_cost_complexity_pruning` for details.
    
- ```max_samples : int or float, default=None```

    If bootstrap is True, the number of samples to draw from X to train each base estimator.
    - If None (default), then draw `X.shape[0]` samples.
    - If int, then draw `max_samples` samples.
    - If float, then draw `max_samples * X.shape[0]` samples. Thus,
      `max_samples` should be in the interval `(0, 1]`.

#### Attributes:

- ```n_features_in_ : int```

    The number of features when :meth:`fit` is performed.
    
- ```feature_importances_ : ndarray of shape (n_features, )```

    The impurity-based feature importances.
    The higher, the more important the feature.
    The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature.  It is also known as the Gini importance.
    
- ```coef_ : array of shape (n_features, ) or (n_targets, n_features)```

    Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.
    
- ```intercept_ : float or array of shape (n_targets,)```

    Independent term in the linear model. Set to 0 if `fit_intercept = False` in `base_estimator`. 
    
- ```base_estimator_ : object```

    A fitted linear model instance.
    
- ```forest_estimator_ : object```

    A fitted random forest instance. 
    
#### Methods:

- ```fit(X, y, sample_weight=None)```

    Build a Linear Forest from the training set (X, y).
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        The training input samples.  
    
    - `y` : array-like of shape (n_samples, ) or (n_samples, n_targets)
        
        Target values.
    
    - `sample_weight` : array-like of shape (n_samples, ), default=None
        
        Sample weights. 
        
    **Returns:**
    
    - `self` : object

- ```predict(X)```

    Predict regression target for X.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        Samples.  
        
    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, ) or also (n_samples, n_targets) if multitarget regression.
    
        The predicted values.
        
- ```apply(X)```

    Apply trees in the forest to X, return leaf indices.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        The input samples.  
        
    **Returns:**
    
    - `X_leaves` : array-like of shape (n_samples, n_estimators).
    
        For each datapoint x in X and for each tree in the forest, return the index of the leaf x ends up in.
        
- ```decision_path(X)```

    Return the decision path in the forest.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        The input samples.  
        
    **Returns:**
    
    - `indicator` : sparse matrix of shape (n_samples, n_nodes)
        
        Return a node indicator matrix where non zero elements indicates that the samples goes through the nodes. The matrix is of CSR format.
        
    - `n_nodes_ptr` : ndarray of shape (n_estimators + 1, )
        
        The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]] gives the indicator value for the i-th estimator.

## LinearForestClassifier
```
class lineartree.LinearForestClassifier(base_estimator, *, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., bootstrap=True, oob_score=False, n_jobs=None, random_state=None, ccp_alpha=0.0, max_samples=None)
```

#### Parameters:

- ```base_estimator : object```

    The linear estimator fitted on the raw target.
    The linear estimator must be a regressor from sklearn.linear_model.
    
- ```n_estimators : int, default=100```

    The number of trees in the forest.
    
- ```max_depth : int, default=None```

    The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    
- ```min_samples_split : int or float, default=2```

    The minimum number of samples required to split an internal node:    
    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.
      
- ```min_samples_leaf : int or float, default=1```

    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.
      
- ```min_weight_fraction_leaf : float, default=0.0```

    The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
    
- ```max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"```

    The number of features to consider when looking for the best split:    
    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a fraction and `round(max_features * n_features)` features are considered at each split.
    - If "auto", then `max_features=n_features`.
    - If "sqrt", then `max_features=sqrt(n_features)`.
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.
    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.
    
- ```max_leaf_nodes : int, default=None```

    Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    
- ```min_impurity_decrease : float, default=0.0```

    A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    
- ```bootstrap : bool, default=True```

    Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
    
- ```oob_score : bool, default=False```

    Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True.
    
- ```n_jobs : int, default=None```

    The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`, :meth:`decision_path` and :meth:`apply` are all parallelized over the trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
    
- ```random_state : int, RandomState instance or None, default=None```

    Controls both the randomness of the bootstrapping of the samples used when building trees (if ``bootstrap=True``) and the sampling of the features to consider when looking for the best split at each node (if ``max_features < n_features``).
    
- ```ccp_alpha : non-negative float, default=0.0```

    Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ``ccp_alpha`` will be chosen. By default, no pruning is performed. See :ref:`minimal_cost_complexity_pruning` for details.
    
- ```max_samples : int or float, default=None```

    If bootstrap is True, the number of samples to draw from X to train each base estimator.
    - If None (default), then draw `X.shape[0]` samples.
    - If int, then draw `max_samples` samples.
    - If float, then draw `max_samples * X.shape[0]` samples. Thus,
      `max_samples` should be in the interval `(0, 1]`.

#### Attributes:

- ```n_features_in_ : int```

    The number of features when :meth:`fit` is performed.
    
- ```feature_importances_ : ndarray of shape (n_features, )```

    The impurity-based feature importances.
    The higher, the more important the feature.
    The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature.  It is also known as the Gini importance.
    
- ```coef_ : array of shape (n_features, ) or (n_targets, n_features)```

    Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.
    
- ```intercept_ : float or array of shape (n_targets,)```

    Independent term in the linear model. Set to 0 if `fit_intercept = False` in `base_estimator`. 
    
- ```classes_ : ndarray of shape (n_classes, )```

    A list of class labels known to the classifier. 
    
- ```base_estimator_ : object```

    A fitted linear model instance.
    
- ```forest_estimator_ : object```

    A fitted random forest instance. 
    
#### Methods:

- ```fit(X, y, sample_weight=None)```

    Build a Linear Forest from the training set (X, y).
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        The training input samples.  
    
    - `y` : array-like of shape (n_samples, ) or (n_samples, n_targets)
        
        Target values.
    
    - `sample_weight` : array-like of shape (n_samples, ), default=None
        
        Sample weights. 
        
    **Returns:**
    
    - `self` : 
    
- ```decision_function(X)```

    Predict confidence scores for samples.
    The confidence score for a sample is proportional to the signed distance of that sample to the hyperplane.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        Samples.  
        
    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, ).
    
        Confidence scores.
        Confidence score for self.classes_[1] where >0 means this class would be predicted.

- ```predict(X)```

    Predict class for X.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        Samples.  
        
    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, ).
    
        The predicted classes.
        
- ```predict_proba(X)```

    Predict class probabilities for X.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        Samples.  
        
    **Returns:**
    
    - `proba` : ndarray of shape (n_samples, n_classes).
    
        The class probabilities of the input samples. The order of the classes corresponds to that in the attribute :term:`classes_`.
        
- ```predict_log_proba(X)```

    Predict class log-probabilities for X.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        Samples.  
        
    **Returns:**
    
    - `pred` : ndarray of shape (n_samples, n_classes).
    
        The class log-probabilities of the input samples. The order of the classes corresponds to that in the attribute :term:`classes_`.
        
- ```apply(X)```

    Apply trees in the forest to X, return leaf indices.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        The input samples.  
        
    **Returns:**
    
    - `X_leaves` : array-like of shape (n_samples, n_estimators).
    
        For each datapoint x in X and for each tree in the forest, return the index of the leaf x ends up in.
        
- ```decision_path(X)```

    Return the decision path in the forest.
    
    **Parameters:**
    
    - `X` : array-like of shape (n_samples, n_features)
        
        The input samples.  
        
    **Returns:**
    
    - `indicator` : sparse matrix of shape (n_samples, n_nodes)
        
        Return a node indicator matrix where non zero elements indicates that the samples goes through the nodes. The matrix is of CSR format.
        
    - `n_nodes_ptr` : ndarray of shape (n_estimators + 1, )
        
        The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]] gives the indicator value for the i-th estimator.