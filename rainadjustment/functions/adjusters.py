# -*- coding: utf-8 -*-
"""
Functions to adjust the gridded rainfall observations / estimates with rain
gauge observations.

Available functions:
- apply_adjustment
- check_adjustment_factor
- __obtain_adjustment_method
- __kriging_adjustment
"""

import numpy as np
import skgstat as skg
import wradlib as wrl

from utils.utils import get_rawatobs, get_interpolation_method


def apply_adjustment(config_xml, obs_coords, obs_values, grid_coords, grid_values, logger):
    """
    Main function to apply the adjustment on the gridded rainfall product.

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
    logger: logging instance
        Logger for log messages, passed on from the main.py script.


    Returns
    ------
    adjusted_values: ndarray(float)
        The adjusted gridded rainfall values in the same shape as
        grid_values.
    """
    if config_xml["adjustment_method"] == "KED":
        return __kriging_adjustment(
            config_xml=config_xml,
            obs_coords=obs_coords,
            obs_values=obs_values,
            grid_coords=grid_coords,
            grid_values=grid_values,
            logger=logger,
        )
    else:
        adjuster = __obtain_adjustment_method(
            config_xml=config_xml, obs_coords=obs_coords, grid_coords=grid_coords, logger=logger
        )
        return adjuster(obs_values, grid_values)


def check_adjustment_factor(
    adjusted_values,
    original_values,
    max_change_factor,
):
    """
    Parameters
    ----------
    adjusted_values: ndarray(float)
        List of floats containing the adjusted gridded rainfall values.
    original_values: ndarray(float)
        List of floats containing the orginal gridded rainfall values.
    max_change_factor: float | None
        Maximum change (both increase and decrease) of the adjusted
        gridded rainfall per grid cell point. The values should be provided
        as float of the factor (e.g. 2.0 - max. two times smaller or bigger
        than orginal).

    Returns
    -------
    checked_adjusted_values: ndarray(float)
        The adjuster that is called, already initialized for the obs and
        grid coords. A check if sufficient valid observation-grid pairs
        are present has already taken place by wradlib.
    """
    # Check the factor increase and decrease from the original_values to the
    # adjusted_values
    factor_change = adjusted_values / original_values

    # Where factor_change is larger than max_change_factor or smaller than
    # 1/max_change_factor, we adjust the original value with the provide
    # max_change_factor (or 1/max_change_factor) to ensure that the correction
    # is not blowing up.
    if max_change_factor is not None:
        conditions = [
            factor_change > max_change_factor,
            factor_change < (1 / max_change_factor),
        ]
        choices = [
            original_values * max_change_factor,
            original_values / max_change_factor,
        ]

        checked_adjusted_values = np.select(conditions, choices, default=adjusted_values)
    else:
        checked_adjusted_values = adjusted_values

    return checked_adjusted_values


def __obtain_adjustment_method(config_xml, obs_coords, grid_coords, logger):
    """
    Parameters
    ----------
    config_xml: dict
        Dictionary containing the adjustment settings.
    obs_coords: list(float)
        List of float containing the latitude and longitude values of the
        gauges as (lat, lon).
    grid_coords: ndarray(float)
        List of floats containing the latitude and longitude values of the
        gridded rainfall as (lat, lon).
    logger: logging instance
        Logger for log messages, passed on from the main.py script.


    Returns
    ------
    adjuster: wrl instance
        The adjuster that is called, already initialized for the obs and
        grid coords. A check if sufficient valid observation-grid pairs
        are present has already taken place by wradlib.
    """
    adjustment_method = config_xml["adjustment_method"]
    interpolation_method = get_interpolation_method(config_xml)

    if adjustment_method == "MFB":
        return wrl.adjust.AdjustMFB(
            obs_coords=obs_coords,
            raw_coords=grid_coords,
            nnear_raws=config_xml["nearest_cells_to_use"],
            stat=config_xml["statistical_function"],
            mingages=config_xml["min_gauges"],
            minval=config_xml["threshold"],
            mfb_args={"method": "median"},
        )
    elif adjustment_method == "Additive":
        return wrl.adjust.AdjustBase(
            obs_coords=obs_coords,
            raw_coords=grid_coords,
            nnear_raws=config_xml["nearest_cells_to_use"],
            stat=config_xml["statistical_function"],
            mingages=config_xml["min_gauges"],
            minval=config_xml["threshold"],
            ipclass=interpolation_method,
        )
    elif adjustment_method == "Multiplicative":
        return wrl.adjust.AdjustMultiply(
            obs_coords=obs_coords,
            raw_coords=grid_coords,
            nnear_raws=config_xml["nearest_cells_to_use"],
            stat=config_xml["statistical_function"],
            mingages=config_xml["min_gauges"],
            minval=config_xml["threshold"],
            ipclass=interpolation_method,
        )
    elif adjustment_method == "Mixed":
        return wrl.adjust.AdjustMixed(
            obs_coords=obs_coords,
            raw_coords=grid_coords,
            nnear_raws=config_xml["nearest_cells_to_use"],
            stat=config_xml["statistical_function"],
            mingages=config_xml["min_gauges"],
            minval=config_xml["threshold"],
            ipclass=interpolation_method,
        )
    else:
        logger.error(
            "Requested adjustment method %s not available, choose from ['MFB', 'Additive', 'Multiplicative', 'Mixed' and 'KED']",
            adjustment_method,
        )
        raise KeyError(
            "Requested adjustment method %s not available, choose from ['MFB', 'Additive', 'Multiplicative', 'Mixed' and 'KED']",
            adjustment_method,
        )


def __kriging_adjustment(config_xml, obs_coords, obs_values, grid_coords, grid_values, logger):
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
    logger: logging instance
        Logger for log messages, passed on from the main.py script.


    Returns
    ------
    adjusted_values: ndarray(float)
        The adjusted gridded rainfall values in the same shape as grid_values.
    """
    # First, find the grid_values that correspond to the matching obs_coords
    rawatobs, ix = get_rawatobs(
        config_xml=config_xml,
        obs_coords=obs_coords,
        obs_values=obs_values,
        grid_coords=grid_coords,
        grid_values=grid_values,
    )

    # Determine the variogram
    if config_xml["variogram_model"] == "standard":
        semivariogram = "1.0 Exp(10000.)"
    elif config_xml["variogram_model"] == "auto_derive":
        try:
            skg_variogram = skg.Variogram(
                obs_coords[ix],
                obs_values[ix],
                maxlag="median",
                model="spherical",
                n_lags=10,
                normalize=False,
                use_nugget=True,
            )
            semivariogram = f"1.0 Nug({skg_variogram.describe()['nugget']}) + {skg_variogram.describe()['sill']} Sph({skg_variogram.describe()['effective_range']})"
        except (AttributeError, ValueError, RuntimeError) as e:
            logger.info(
                "Not able to derive the Variogram, we'll continue with the default value of 1.0 Exp(10000.)"
            )
            semivariogram = "1.0 Exp(10000.)"
    else:
        semivariogram = config_xml["variogram_model"]

    if rawatobs is not None:
        # Get the kriging method
        if config_xml["adjustment_method"] == "KED":
            kriging_method = wrl.ipol.ExternalDriftKriging(
                obs_coords[ix],
                grid_coords,
                src_drift=rawatobs[ix],
                trg_drift=grid_values,
                cov=semivariogram,
                nnearest=config_xml["kriging_n_gauges_to_use"],
            )
            # Apply the method
            return kriging_method(obs_values[ix])
    return grid_values
