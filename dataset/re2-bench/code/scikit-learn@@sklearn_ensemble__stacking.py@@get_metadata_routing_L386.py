from abc import ABCMeta, abstractmethod
from numbers import Integral
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)

class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble, metaclass=ABCMeta):
    """Base class for stacking method."""
    _parameter_constraints: dict = {'estimators': [list], 'final_estimator': [None, HasMethods('fit')], 'cv': ['cv_object', StrOptions({'prefit'})], 'n_jobs': [None, Integral], 'passthrough': ['boolean'], 'verbose': ['verbose']}

    @abstractmethod
    def __init__(self, estimators, final_estimator=None, *, cv=None, stack_method='auto', n_jobs=None, verbose=0, passthrough=False):
        super().__init__(estimators=estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.6

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self)
        for name, estimator in self.estimators:
            router.add(**{name: estimator}, method_mapping=MethodMapping().add(callee='fit', caller='fit'))
        try:
            final_estimator_ = self.final_estimator_
        except AttributeError:
            final_estimator_ = self.final_estimator
        router.add(final_estimator_=final_estimator_, method_mapping=MethodMapping().add(caller='predict', callee='predict'))
        return router
