from functools import partial
from typing import Protocol, Any, runtime_checkable

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


class _Estimator(BaseEstimator):

    def __init__(self, estimator: BaseEstimator, label_transformer: BaseTransformer) -> None:
        super().__init__()

        self._label_transformer = label_transformer
        self._estimator = estimator

    def predict(self, X: Any, **kwargs) -> Any:
        y = self._estimator.predict(X, **kwargs)
        return self._label_transformer.inverse_transform(y)


# Supervised Estimator
def _BestEstimatorFactory(X: Any, y: Any, estimator: EstimatorMixin = SVC(), **kwargs) -> object:
    encoder = LabelEncoder()
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
