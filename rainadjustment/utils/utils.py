# -*- coding: utf-8 -*-
"""
Utility functions for pyRainAdjustment.

Available functons:
- get_interpolation_method
- interpolate
- get_rawatobs
- __idvalid
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import wradlib as wrl


def get_interpolation_method(config_xml: dict[str, Any], logger: logging.Logger):
    """
    Parameters
    ----------
    config_xml: dict
        Dictionary containing the adjustment settings.

    Returns
    ------
    interpolation_methods: callable interpolation method function from wrl.ipol
    """
    interpolation_methods = dict()
    interpolation_methods["Nearest"] = wrl.ipol.Nearest
    interpolation_methods["Idw"] = wrl.ipol.Idw
    interpolation_methods["Linear"] = wrl.ipol.Linear
    interpolation_methods["OrdinaryKriging"] = wrl.ipol.OrdinaryKriging
    interpolation_methods["ExternalDriftKriging"] = wrl.ipol.ExternalDriftKriging

    try:
        return interpolation_methods[config_xml["interpolation_method"]]
    except KeyError as exc:
        logger.error(
            "Unknown interpolation method %s. The available methods are: %s",
            config_xml["interpolation_method"],
            str(list(interpolation_methods.keys())),
        )
        raise ValueError(
            "Unknown interpolation method {}\n".format(config_xml["interpolation_method"])
            + "The available methods are:"
            + str(list(interpolation_methods.keys()))
        ) from exc


def interpolate(
    interpolation_method,
    obs_coords: npt.NDArray[np.float64],
    grid_coords: npt.NDArray[np.float64],
    values_to_interpolate: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Parameters
    ----------
    interpolation_method: wrl.ipol method
        interpolation_method: callable interpolation method function from wrl.ipol.
        The method can be obtained from the get_interpolation_method function.
    obs_coords: ndarray(float)
        List of floats containing the latitude and longitude values of the
        gauges as (lat, lon).
    grid_coords: ndarray(float)
        List of floats containing the latitude and longitude values of the
        gridded rainfall as (lat, lon).
    values_to_interpolate: ndarray(float)
        List of floats containing the values to interpolate on the grid_coords. There
        should be one value per (lat, lon) coordinate in obs_coords.

    Returns
    ------
    interpolated_values: ndarray(float)
            The interpolated values on the location of the grid_coords.
    """
    # Get the interpolator instance
    interpolator = interpolation_method(
        obs_coords,
        grid_coords,
    )
    # Apply the method
    return interpolator(values_to_interpolate)


def get_rawatobs(
    config_xml: dict[str, Any],
    obs_coords: npt.NDArray[np.float64],
    obs_values: npt.NDArray[np.float64],
    grid_coords: npt.NDArray[np.float64],
    grid_values: npt.NDArray[np.float64],
) -> tuple[None | npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Parameters
    ----------
    config_xml: dict
        Dictionary containing the adjustment settings.
    obs_coords: ndarray(float)
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


def __idvalid(
    data: npt.NDArray[np.float64],
    isinvalid: list[float] | None = None,
    minval: float | None = None,
    maxval: float | None = None,
) -> npt.NDArray[np.int16]:
    """Function from wradlib: identifies valid entries in an array and returns
    the corresponding indices.

    Invalid values are NaN and Inf. Other invalid values can be passed using
    the isinvalid keyword argument.

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray`
    isinvalid : list | None
        List of what is considered an invalid value
    minval: float | None
        Minimum value to be used.
    maxval: float | None
        Maximum value to be used.

    Returns
    ------
    valid entries: ndarray(int)
        Indices of the valid entries.
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
