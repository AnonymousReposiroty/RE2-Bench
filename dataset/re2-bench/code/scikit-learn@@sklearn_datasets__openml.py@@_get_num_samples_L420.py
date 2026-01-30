def _get_num_samples(data_qualities: OpenmlQualitiesType) -> int:
    """Get the number of samples from data qualities.

    Parameters
    ----------
    data_qualities : list of dict
        Used to retrieve the number of instances (samples) in the dataset.

    Returns
    -------
    n_samples : int
        The number of samples in the dataset or -1 if data qualities are
        unavailable.
    """
    # If the data qualities are unavailable, we return -1
    default_n_samples = -1

    qualities = {d["name"]: d["value"] for d in data_qualities}
    return int(float(qualities.get("NumberOfInstances", default_n_samples)))
