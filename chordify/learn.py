from functools import partial
from typing import Protocol, Any, runtime_checkable

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@runtime_checkable
class BaseTransformer(Protocol):

    def fit(self, X: Any) -> Any: ...

    def transform(self, X: Any) -> Any: ...

    def fit_transform(self, X: Any) -> Any: ...

    def inverse_transform(self, y: Any) -> Any: ...


@runtime_checkable
class BaseEstimator(Protocol):

    def predict(self, X: Any, **kwargs) -> Any: ...


@runtime_checkable
class EstimatorMixin(BaseEstimator, Protocol):

    # Supervised Estimator
    def fit(self, X: Any, y: Any, **kwargs) -> Any: ...


# preserves rules from chord recognition module
@runtime_checkable
class PredictStrategy(Protocol):
    def predict(self, frame: Any, *args) -> Any:
        ...


# preserves rules from chord recognition module
class _Estimator(PredictStrategy):

    def __init__(self, estimator: BaseEstimator, label_transformer: BaseTransformer) -> None:
        super().__init__()

        self._label_transformer = label_transformer
        self._estimator = estimator

    def predict(self, frame: Any, *args) -> Any:
        # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or
        # array.reshape(1, -1) if it contains a single sample.

        y = self._estimator.predict(numpy.asarray(frame).reshape(1, -1))
        return self._label_transformer.inverse_transform(y)[0]


class _ObjectEncoder(LabelEncoder):
    """ Encode objects instead of numbers or strings """
    _classes = None

    def fit(self, y):
        self._classes = dict(map(lambda obj: (str(obj), obj), y))
        keys = tuple(map(lambda obj: str(obj), y))
        return super().fit(keys)

    def transform(self, y):
        keys = tuple(map(lambda obj: str(obj), y))
        return super().transform(keys)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        keys = super().inverse_transform(y)
        values = tuple(map(lambda obj: self._classes[obj], keys))

        return values


# Supervised Estimator
def _BestEstimatorFactory(X: Any, y: Any, estimator: EstimatorMixin = SVC(), **kwargs) -> PredictStrategy:
    """ Be sure that y argument can be encoded and take note that it will be output of estimator """
    encoder = _ObjectEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    grid_search = GridSearchCV(estimator=estimator, param_grid=kwargs, n_jobs=-1)
    grid_search.fit(X, encoded_y)

    return _Estimator(grid_search.best_estimator_, encoder)


SVMClassifier = partial(
    _BestEstimatorFactory,
    estimator=SVC(),
    kernel=['sigmoid', 'rbf', 'poly', 'linear'],
    C=list(range(1, 100, 10)),
    degree=list(range(3, 10))
)

TreeClassifier = partial(
    _BestEstimatorFactory,
    estimator=DecisionTreeClassifier(),
    max_depth=list(range(3, 20))
)

KNeighborsClassifier = partial(
    _BestEstimatorFactory,
    estimator=KNeighborsClassifier(),
    n_neighbors=list(range(24, 48))
)

RandomForestClassifier = partial(
    _BestEstimatorFactory,
    estimator=RandomForestClassifier(),
    n_estimators=list(range(10, 200, 10))
)
