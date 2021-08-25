import numpy as np


SCORING = {
    'linear': lambda y, yh: y - yh,
    'square': lambda y, yh: np.square(y - yh),
    'absolute': lambda y, yh: np.abs(y - yh),
    'exponential': lambda y, yh: 1 - np.exp(-np.abs(y - yh)),
    'poisson': lambda y, yh: yh.clip(1e-6) - y * np.log(yh.clip(1e-6)),
    'hamming': lambda y, yh, classes: (y != yh).astype(int),
    'entropy': lambda y, yh, classes: np.sum(list(map(
        lambda c: -(y == c[1]).astype(int) * np.log(yh[:, c[0]]),
        enumerate(classes))), axis=0)
}


def _normalize_score(scores, weights=None):
    """Normalize scores according to weights"""

    if weights is None:
        return scores.mean()
    else:
        return np.mean(np.dot(scores.T, weights) / weights.sum())


def mse(model, X, y, weights=None, **largs):
    """Mean Squared Error"""

    pred = model.predict(X)
    scores = SCORING['square'](y, pred)

    return _normalize_score(scores, weights)


def rmse(model, X, y, weights=None, **largs):
    """Root Mean Squared Error"""

    return np.sqrt(mse(model, X, y, weights, **largs))


def mae(model, X, y, weights=None, **largs):
    """Mean Absolute Error"""

    pred = model.predict(X)
    scores = SCORING['absolute'](y, pred)

    return _normalize_score(scores, weights)


def poisson(model, X, y, weights=None, **largs):
    """Poisson Loss"""

    if np.any(y < 0):
        raise ValueError("Some value(s) of y are negative which is"
                         " not allowed for Poisson regression.")

    pred = model.predict(X)
    scores = SCORING['poisson'](y, pred)

    return _normalize_score(scores, weights)


def hamming(model, X, y, weights=None, **largs):
    """Hamming Loss"""

    pred = model.predict(X)
    scores = SCORING['hamming'](y, pred, None)

    return _normalize_score(scores, weights)


def crossentropy(model, X, y, classes, weights=None, **largs):
    """Cross Entropy Loss"""

    pred = model.predict_proba(X).clip(1e-5, 1 - 1e-5)
    scores = SCORING['entropy'](y, pred, classes)

    return _normalize_score(scores, weights)