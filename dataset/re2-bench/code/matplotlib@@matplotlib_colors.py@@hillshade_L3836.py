import numpy as np

class LightSource:
    """
    Create a light source coming from the specified azimuth and elevation.
    Angles are in degrees, with the azimuth measured
    clockwise from north and elevation up from the zero plane of the surface.

    `shade` is used to produce "shaded" RGB values for a data array.
    `shade_rgb` can be used to combine an RGB image with an elevation map.
    `hillshade` produces an illumination map of a surface.
    """

    def __init__(self, azdeg=315, altdeg=45, hsv_min_val=0, hsv_max_val=1, hsv_min_sat=1, hsv_max_sat=0):
        """
        Specify the azimuth (measured clockwise from south) and altitude
        (measured up from the plane of the surface) of the light source
        in degrees.

        Parameters
        ----------
        azdeg : float, default: 315 degrees (from the northwest)
            The azimuth (0-360, degrees clockwise from North) of the light
            source.
        altdeg : float, default: 45 degrees
            The altitude (0-90, degrees up from horizontal) of the light
            source.
        hsv_min_val : number, default: 0
            The minimum value ("v" in "hsv") that the *intensity* map can shift the
            output image to.
        hsv_max_val : number, default: 1
            The maximum value ("v" in "hsv") that the *intensity* map can shift the
            output image to.
        hsv_min_sat : number, default: 1
            The minimum saturation value that the *intensity* map can shift the output
            image to.
        hsv_max_sat : number, default: 0
            The maximum saturation value that the *intensity* map can shift the output
            image to.

        Notes
        -----
        For backwards compatibility, the parameters *hsv_min_val*,
        *hsv_max_val*, *hsv_min_sat*, and *hsv_max_sat* may be supplied at
        initialization as well.  However, these parameters will only be used if
        "blend_mode='hsv'" is passed into `shade` or `shade_rgb`.
        See the documentation for `blend_hsv` for more details.
        """
        self.azdeg = azdeg
        self.altdeg = altdeg
        self.hsv_min_val = hsv_min_val
        self.hsv_max_val = hsv_max_val
        self.hsv_min_sat = hsv_min_sat
        self.hsv_max_sat = hsv_max_sat

    @property
    def direction(self):
        """The unit vector direction towards the light source."""
        az = np.radians(90 - self.azdeg)
        alt = np.radians(self.altdeg)
        return np.array([np.cos(az) * np.cos(alt), np.sin(az) * np.cos(alt), np.sin(alt)])

    def hillshade(self, elevation, vert_exag=1, dx=1, dy=1, fraction=1.0):
        """
        Calculate the illumination intensity for a surface using the defined
        azimuth and elevation for the light source.

        This computes the normal vectors for the surface, and then passes them
        on to `shade_normals`

        Parameters
        ----------
        elevation : 2D array-like
            The height values used to generate an illumination map
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs. meters) or to
            exaggerate or de-emphasize topographic effects.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.

        Returns
        -------
        `~numpy.ndarray`
            A 2D array of illumination values between 0-1, where 0 is
            completely in shadow and 1 is completely illuminated.
        """
        dy = -dy
        e_dy, e_dx = np.gradient(vert_exag * elevation, dy, dx)
        normal = np.empty(elevation.shape + (3,)).view(type(elevation))
        normal[..., 0] = -e_dx
        normal[..., 1] = -e_dy
        normal[..., 2] = 1
        normal /= _vector_magnitude(normal)
        return self.shade_normals(normal, fraction)

    def shade_normals(self, normals, fraction=1.0):
        """
        Calculate the illumination intensity for the normal vectors of a
        surface using the defined azimuth and elevation for the light source.

        Imagine an artificial sun placed at infinity in some azimuth and
        elevation position illuminating our surface. The parts of the surface
        that slope toward the sun should brighten while those sides facing away
        should become darker.

        Parameters
        ----------
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.

        Returns
        -------
        `~numpy.ndarray`
            A 2D array of illumination values between 0-1, where 0 is
            completely in shadow and 1 is completely illuminated.
        """
        intensity = normals.dot(self.direction)
        imin, imax = (intensity.min(), intensity.max())
        intensity *= fraction
        if imax - imin > 1e-06:
            intensity -= imin
            intensity /= imax - imin
        intensity = np.clip(intensity, 0, 1)
        return intensity
