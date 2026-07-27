from numbers import Integral
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    clone,
)
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)

class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
    """Base class for voting.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _parameter_constraints: dict = {'estimators': [list], 'weights': ['array-like', None], 'n_jobs': [None, Integral], 'verbose': ['verbose']}

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
        for name, estimator in self.estimators:
            router.add(**{name: estimator}, method_mapping=MethodMapping().add(callee='fit', caller='fit'))
        return router
