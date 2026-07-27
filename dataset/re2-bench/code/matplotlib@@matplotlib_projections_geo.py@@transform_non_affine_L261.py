import numpy as np

class AitoffTransform(_GeoTransform):
    """The base Aitoff transform."""

    def transform_non_affine(self, values):
        longitude, latitude = values.T
        half_long = longitude / 2.0
        cos_latitude = np.cos(latitude)
        alpha = np.arccos(cos_latitude * np.cos(half_long))
        sinc_alpha = np.sinc(alpha / np.pi)
        x = cos_latitude * np.sin(half_long) / sinc_alpha
        y = np.sin(latitude) / sinc_alpha
        return np.column_stack([x, y])
