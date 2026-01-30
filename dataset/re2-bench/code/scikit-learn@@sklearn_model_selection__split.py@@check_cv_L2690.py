import numbers
from collections.abc import Iterable
from sklearn.utils.multiclass import type_of_target

def check_cv(cv=5, y=None, *, classifier=False):
    """Input checker utility for building a cross-validator.

    Parameters
    ----------
    cv : int, cross-validation generator, iterable or None, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value changed from 3-fold to 5-fold.

    y : array-like, default=None
        The target variable for supervised learning problems.

    classifier : bool, default=False
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.

    Examples
    --------
    >>> from sklearn.model_selection import check_cv
    >>> check_cv(cv=5, y=None, classifier=False)
    KFold(...)
    >>> check_cv(cv=5, y=[1, 1, 0, 0, 0, 0], classifier=True)
    StratifiedKFold(...)
    """
    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral):
        if (
            classifier
            and (y is not None)
            and (type_of_target(y, input_name="y") in ("binary", "multiclass"))
        ):
            return StratifiedKFold(cv)
        else:
            return KFold(cv)

    if not hasattr(cv, "split") or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError(
                "Expected cv as an integer, cross-validation "
                "object (from sklearn.model_selection) "
                "or an iterable. Got %s." % cv
            )
        return _CVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification
