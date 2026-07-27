def _tolist(array, xp=None):
    xp, _ = get_namespace(array, xp=xp)
    if _is_numpy_namespace(xp):
        return array.tolist()
    array_np = _convert_to_numpy(array, xp=xp)
    return [element.item() for element in array_np]
