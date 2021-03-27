import numpy as np


def _normalize_score(scores, weights=None):
    """Normalize scores according to weights"""
    if weights is None:
        return scores.mean()
    else:
        return np.mean(np.dot(scores.T, weights) / weights.sum())


def mse(model, X, y, weights=None):
    """Mean Squared Error"""
    pred = model.predict(X)
    scores = np.square(y - pred)
    
    return _normalize_score(scores, weights)

def rmse(model, X, y, weights=None):
    """Root Mean Squared Error"""
    return np.sqrt(mse(model, X, y, weights))
    

def mae(model, X, y, weights=None): 
    """Mean Aquared Error"""
    pred = model.predict(X)
    scores = np.abs(y - pred)
    
    return _normalize_score(scores, weights)


def poisson(model, X, y, weights=None):
    """Poisson Loss"""
    # from LightGBM implementation:
    # poiss_loss = pred - y * log(pred)
    # pred_exp = exp(pred)
    # poiss_loss = pred_exp - y * pred
    
    if np.any(y < 0):
        raise ValueError("Some value(s) of y are negative which is"
                         " not allowed for Poisson regression.")
    
    pred = np.exp(model.predict(X))
    scores = pred - y * np.log(pred)
    
    return _normalize_score(scores, weights) 


def hamming(model, X, y, classes=None, weights=None):
    """Hamming Loss"""
    # classes param exist only for compatibility w/ crossentropy
    
    pred = model.predict(X)
    scores = (y != pred).astype(int)
    
    return _normalize_score(scores, weights)
        
    
def crossentropy(model, X, y, classes, weights=None):
    """Cross Entropy Loss"""
    pred = model.predict_proba(X)
    pred = pred.clip(1e-5, 1 - 1e-5)
    
    # loop in each class to limit memory spikes
    entropy = 0.
    for c,cls in enumerate(classes):
        y_cls = (y == cls).astype(int)
        scores = - (y_cls * np.log(pred[:,c]))
        entropy += _normalize_score(scores, weights)
    
    return entropy