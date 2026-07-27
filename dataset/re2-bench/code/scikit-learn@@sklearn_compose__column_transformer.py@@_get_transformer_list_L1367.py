from sklearn.pipeline import _fit_transform_one, _name_estimators, _transform_one

def _get_transformer_list(estimators):
    """
    Construct (name, trans, column) tuples from list

    """
    transformers, columns = zip(*estimators)
    names, _ = zip(*_name_estimators(transformers))

    transformer_list = list(zip(names, transformers, columns))
    return transformer_list
