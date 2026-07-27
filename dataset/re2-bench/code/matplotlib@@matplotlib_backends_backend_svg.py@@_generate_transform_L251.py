from matplotlib.transforms import Affine2D, Affine2DBase

def _generate_transform(transform_list):
    parts = []
    for type, value in transform_list:
        if (type == 'scale' and (value == (1,) or value == (1, 1))
                or type == 'translate' and value == (0, 0)
                or type == 'rotate' and value == (0,)):
            continue
        if type == 'matrix' and isinstance(value, Affine2DBase):
            value = value.to_values()
        parts.append('{}({})'.format(
            type, ' '.join(_short_float_fmt(x) for x in value)))
    return ' '.join(parts)
