import numpy as np

def _update_filter_sdas(sdas, mib, xi_complement, reachability_plot):
    """Update steep down areas (SDAs) using the new maximum in between (mib)
    value, and the given complement of xi, i.e. ``1 - xi``.
    """
    if np.isinf(mib):
        return []
    res = [
        sda for sda in sdas if mib <= reachability_plot[sda["start"]] * xi_complement
    ]
    for sda in res:
        sda["mib"] = max(sda["mib"], mib)
    return res
