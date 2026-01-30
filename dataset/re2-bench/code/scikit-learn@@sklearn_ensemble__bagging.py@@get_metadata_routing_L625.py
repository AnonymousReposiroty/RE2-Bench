from abc import ABCMeta, abstractmethod
from numbers import Integral
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.utils._param_validation import HasMethods, Interval, RealNotInt
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)

class BaseBagging(BaseEnsemble, metaclass=ABCMeta):
    """Base class for Bagging meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit', 'predict']), None], 'n_estimators': [Interval(Integral, 1, None, closed='left')], 'max_samples': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='right')], 'max_features': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='right')], 'bootstrap': ['boolean'], 'bootstrap_features': ['boolean'], 'oob_score': ['boolean'], 'warm_start': ['boolean'], 'n_jobs': [None, Integral], 'random_state': ['random_state'], 'verbose': ['verbose']}

    @abstractmethod
    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0):
        super().__init__(estimator=estimator, n_estimators=n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self)
        method_mapping = MethodMapping()
        method_mapping.add(caller='fit', callee='fit').add(caller='decision_function', callee='decision_function')
        if hasattr(self._get_estimator(), 'predict_proba'):
            method_mapping.add(caller='predict', callee='predict_proba').add(caller='predict_proba', callee='predict_proba')
        else:
            method_mapping.add(caller='predict', callee='predict').add(caller='predict_proba', callee='predict')
        if hasattr(self._get_estimator(), 'predict_log_proba'):
            method_mapping.add(caller='predict_log_proba', callee='predict_log_proba')
        elif hasattr(self._get_estimator(), 'predict_proba'):
            method_mapping.add(caller='predict_log_proba', callee='predict_proba')
        else:
            method_mapping.add(caller='predict_log_proba', callee='predict')
        router.add(estimator=self._get_estimator(), method_mapping=method_mapping)
        return router

    @abstractmethod
    def _get_estimator(self):
        """Resolve which estimator to return."""
