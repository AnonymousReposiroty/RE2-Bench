from numbers import Integral, Real
from sklearn.linear_model._base import (
    LinearClassifierMixin,
    LinearModel,
    _preprocess_data,
    _rescale_data,
)
from sklearn.metrics import check_scoring, get_scorer, get_scorer_names
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)

class _BaseRidgeCV(LinearModel):
    _parameter_constraints: dict = {'alphas': ['array-like', Interval(Real, 0, None, closed='neither')], 'fit_intercept': ['boolean'], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'cv': ['cv_object'], 'gcv_mode': [StrOptions({'auto', 'svd', 'eigen'}), None], 'store_cv_results': ['boolean'], 'alpha_per_target': ['boolean']}

    def __init__(self, alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, cv=None, gcv_mode=None, store_cv_results=False, alpha_per_target=False):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.cv = cv
        self.gcv_mode = gcv_mode
        self.store_cv_results = store_cv_results
        self.alpha_per_target = alpha_per_target

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
        router = MetadataRouter(owner=self).add_self_request(self).add(scorer=self._get_scorer(), method_mapping=MethodMapping().add(caller='fit', callee='score')).add(splitter=self.cv, method_mapping=MethodMapping().add(caller='fit', callee='split'))
        return router

    def _get_scorer(self):
        """Make sure the scorer is weighted if necessary.

        This uses `self._get_scorer_instance()` implemented in child objects to get the
        raw scorer instance of the estimator, which will be ignored if `self.scoring` is
        not None.
        """
        if _routing_enabled() and self.scoring is None:
            return self._get_scorer_instance().set_score_request(sample_weight=True)
        return check_scoring(estimator=self, scoring=self.scoring, allow_none=True)
