# -*- coding: utf-8 -*-
"""
Utility functions for pyRainAdjustment.
"""

import numpy as np
import wradlib as wrl


def get_rawatobs(config_xml, obs_coords, obs_values, grid_coords, grid_values):
    """
    Parameters
    ----------
    config_xml: dict
        Dictionary containing the adjustment settings.
    obs_coords: list(float)
        List of floats containing the latitude and longitude values of the
        gauges as (lat, lon).
    obs_values: ndarray(float)
        List of floats containing the observed rainfall value per (lat, lon)
        coordinate.
    grid_coords: ndarray(float)
        List of floats containing the latitude and longitude values of the
        gridded rainfall as (lat, lon).
    grid_values: ndarray(float)
        List of floats containing the gridded rainfall value per (lat, lon)
        coordinate in the grid.


    Returns
    ------
    rawatobs: None | ndarray(float)
        List of grid values at the gauge observation locations. If the number
        of points is less
        than min_gauges, None is returned.
    ix: ndarray(float)
        Index locations of the gridded rainfall (rawatobs) and observation
        points that comply with the predefined threshold values.
    """
    # radar values at gauge locations
    rawatobs_class = wrl.adjust.RawAtObs(
        obs_coords=obs_coords,
        raw_coords=grid_coords,
        nnear=config_xml["nearest_cells_to_use"],
        stat=config_xml["statistical_function"],
    )
    rawatobs = rawatobs_class.__call__(raw=grid_values, obs=obs_values)
    # check where both gage and radar observations are valid
    ix = np.intersect1d(
        __idvalid(obs_values, minval=config_xml["threshold"]),
        __idvalid(rawatobs, minval=config_xml["threshold"]),
    )

    # Check if sufficient valid gauges are present. If so, return the grid values
    # at the observations location.
    if len(ix) < config_xml["min_gauges"]:
        return None, ix
    return rawatobs, ix


def __idvalid(data, isinvalid=None, minval=None, maxval=None):
    """Identifies valid entries in an array and returns the corresponding
    indices

    Invalid values are NaN and Inf. Other invalid values can be passed using
    the isinvalid keyword argument.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
    isinvalid : list
        list of what is considered an invalid value

    """
    if isinvalid is None:
        isinvalid = [-99.0, 99, -9999.0, -9999]
    ix = np.ma.masked_invalid(data).mask
    for el in isinvalid:
        ix = np.logical_or(ix, np.ma.masked_where(data == el, data).mask)
    if minval is not None:
        ix = np.logical_or(ix, np.ma.masked_less(data, minval).mask)
    if maxval is not None:
        ix = np.logical_or(ix, np.ma.masked_greater(data, maxval).mask)

    return np.where(np.logical_not(ix))[0]
