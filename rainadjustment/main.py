# -*- coding: utf-8 -*-
"""
Main script that handles the gauge - gridded rainfall product
adjustment procedure.
"""

import argparse
from datetime import datetime
import logging
import os
import time

import numpy as np
import wradlib as wrl
import xarray as xr

from adjusters import apply_adjustment, check_adjustment_factor
from downscaling import downscale_gridded_precip
from xml_config_parser import parse_run_xml


# ----------------------------------------------------------------------- #
# Functions
# ----------------------------------------------------------------------- #
def obtain_gauge_information(gauge_folder):
    """
    Parameters
    ----------
    gauge_folder: str
        The folder containing all netCDFs with the rain gauge observations.
        The rain gauge observations are expected to contain "_Gauges.nc"
        in their name.

    Returns
    ------
    obs_coords: list(float)
        List of float containing the latitude and longitude values of the
        gauges as (lat, lon).
    obs_names: list(str)
        List containing the station names.
    obs_values: list(float)
        List containing the station observation values per station.
    """
    # Create the output list
    obs_coords = []
    obs_names = []
    obs_values = []

    # Open the files
    gauges_files = os.listdir(gauge_folder)
    for gauge_file in gauges_files:
        if gauge_file.endswith("_Gauges.nc"):
            ds = xr.open_dataset(os.path.join(gauge_folder, gauge_file))
            precip_gauges = ds.P[-1, :]  # -1 to only get the last hour of values

            # Per station, store the location, station number and the rainfall
            # value
            for station_index in range(precip_gauges.shape[0]):
                # Get the station information per station
                station_lat = precip_gauges[station_index].lat.values
                station_lon = precip_gauges[station_index].lon.values
                station_name = precip_gauges[station_index].station_id.values
                station_value = precip_gauges[station_index].values

                # Append all the values to the output list
                obs_coords.append([float(station_lon), float(station_lat)])
                obs_names.append(str(station_name.tobytes().decode("utf-8").rstrip("\x00")))
                obs_values.append(float(station_value))

    return np.array(obs_coords), np.array(obs_names), np.array(obs_values)


def obtain_gridded_rainfall_information(grid_file, downscale=False, clim_file=None):
    """
    Parameters
    ----------
    grid_file: str
        The netCDF file containing the gridded rainfall information for
        the last hour.
    downscale: bool
        If True, downscale the gridded precipitation using a (monthly) climatology
        file. Defaults to False.
    clim_file: list(str)
        List of filename(s) containing the monthly climatology information.

    Returns
    ------
    grid_coords: ndarray(float)
        List of floats containing the latitude and longitude values of the
        gridded rainfall as (lat, lon).
    grid_values: ndarray(float)
        List containing the rainfall values per grid point as a flattened
        array.
    grid_shape: (int, int)
        The shape of the gridded rainfall product before flattening
        (lons, lats)
    """
    if downscale:
        ds_gridded = downscale_gridded_precip(precip_orig=grid_file, clim_file=clim_file)
    else:
        # Open the gridded rainfall information
        ds_gridded = xr.open_dataset(grid_file)
    precip_gridded = ds_gridded.P[-1, :, :]

    # Get the grid information
    grid_lats = precip_gridded.y.values
    grid_lons = precip_gridded.x.values
    grid_coords = wrl.util.gridaspoints(grid_lats, grid_lons)
    grid_shape = len(grid_lats), len(grid_lons)
    grid_values = np.array(precip_gridded.values).flatten()

    return grid_coords, grid_values, grid_shape


def store_as_netcdf(adjustment_factor, dataset_example, outfile):
    """
    Saves the output adjustment factors to a new NetCDF file.

    Parameters
    ----------
        adjustment_factor: ndarray
            2D array containing the adjustment factors on the
            original gridded rainfall product grid.
        dataset_example: xr DataSet
            The original gridded rainfall xr DataSet which will
            form the blue print for the output dataset.
        outfile: str
            The output file location.

    Returns
    -------
    None
    """
    # Make a dataset out of the array
    output_dataset = xr.Dataset(
        {
            "adjustment_factor": (
                ("time", "y", "x"),
                [adjustment_factor, adjustment_factor],
            )
        },
        coords={
            "time": dataset_example["time"].values,
            "y": dataset_example["y"].values,
            "x": dataset_example["x"].values,
        },
    )

    output_dataset["crs"] = dataset_example["crs"]

    # Make CF compliant and add global attributes
    output_dataset.attrs.update(
        {
            "title": "Adjustment factor to correct gridded rainfall",
            "institution": "Deltares",
            "source": " ",
            "history": f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}: Created",
            "references": "The open-source Python tool wradlib was used for the adjustment factors",
            "Conventions": "CF-1.8",
            "projection": "EPSG:4326",
        }
    )

    output_dataset["y"].attrs.update(
        {
            "axis": "Y",
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        }
    )

    output_dataset["x"].attrs.update(
        {
            "axis": "X",
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        }
    )

    # Add attributes to data variables
    output_dataset["adjustment_factor"].attrs.update(
        {
            "long_name": "Adjustment Factor",
            "standard_name": "adjustment_factor",
            "units": "-",
        }
    )

    # Saving reprojected data
    output_dataset.to_netcdf(outfile)
    output_dataset.close()

    return


# ----------------------------------------------------------------------- #
# The main work
# ----------------------------------------------------------------------- #
def main():
    start = time.perf_counter()
    work_dir = os.getcwd()

    # Initial parser for global arguments
    global_parser = argparse.ArgumentParser(
        description="Utility to adjust a gridded rainfall product using rain gauges",
        add_help=False,
    )
    global_parser.add_argument(
        "--xml_config",
        type=str,
        help="Path to run.xml config file created by Delft-FEWS.",
    )

    # Parse known global args first
    global_args = global_parser.parse_known_args()[0]

    # Set up the logger
    if not os.path.isdir(os.path.join(work_dir, "logs")):
        os.mkdir(os.path.join(work_dir, "logs"))
    logfn = os.path.join(work_dir, "logs", f"log_meteo_rain_gauge_adjustment.txt")
    logging.basicConfig(
        filename=logfn,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    logger.info("Arguments parsed:\n %s", global_args)

    # If --xml_config is provided, parse it and prepare to use its values
    config_xml = {}
    if global_args.xml_config:
        try:
            config_xml = parse_run_xml(global_args.xml_config)
        except Exception as exception:
            logger.exception(exception, exc_info=True)

    # The actual work
    try:
        # 1. Perform some checks
        adjustment_methods = [
            "MFB",
            "Additive",
            "Multiplicative",
            "Mixed",
            "KED",
        ]
        if config_xml["adjustment_method"] not in adjustment_methods:
            logger.error(
                f"Requested adjustment method not present. Select an adjustment method from {adjustment_methods}"
            )
            raise KeyError("Requested adjustment method not present")
        # Make sure the threshold is always 0.0 when the adjustment_method
        # is Additive
        if config_xml["adjustment_method"] == "Additive" and config_xml["threshold"] > 0.0:
            config_xml["threshold"] = 0.0

        # 2. Get the rain gauge information
        obs_coords, obs_names, obs_values = obtain_gauge_information(
            gauge_folder=os.path.join(work_dir, "input")
        )
        logger.info("Rain gauge information read successfully.")

        # 3. obtain gridded rainfall field
        grid_coords, grid_values, grid_shape = obtain_gridded_rainfall_information(
            grid_file=os.path.join(work_dir, "input", "gridded_rainfall.nc")
        )
        logger.info("Gridded rainfall infromation read successfully.")

        # 4. perform correction
        adjusted_values = apply_adjustment(
            config_xml=config_xml,
            obs_coords=obs_coords,
            obs_values=obs_values,
            grid_coords=grid_coords,
            grid_values=grid_values,
        )

        # A final check to ensure that the adjustment has taken place.
        if np.array_equal(adjusted_values, grid_values):
            if np.isfinite(grid_values).any():
                logger.warning(
                    "Adjustment has not taken place. There were too few valid gauge-grid pairs. The original grid values will be returned."
                )
                adjusted_values_checked = adjusted_values * np.nan
            else:
                logger.warning(
                    "Adjustment has not taken place. The gridded rainfall only contains nans. The original grid values will be returned."
                )
                adjusted_values_checked = adjusted_values * np.nan
        else:
            logger.info("Adjustment has taken place successfully.")
            # Also ensure that the correction values have not been too high.
            adjusted_values_checked = check_adjustment_factor(
                adjusted_values=adjusted_values,
                original_values=grid_values,
                max_change_factor=config_xml["max_change_factor"],
            )
            if np.array_equal(adjusted_values_checked, adjusted_values) == False:
                logger.warning(
                    "Some of the adjusted values were above the set maximum adjustment factor."
                )

        # 5. Return corrected precipitation as a stored netCDF in the output folder
        adjustment_factor = adjusted_values_checked / grid_values
        # Make sure there are no negative numbers and no nans in the adjustment
        adjustment_factor = np.where(adjustment_factor < 0.0, 1.0, adjustment_factor)
        adjustment_factor = np.nan_to_num(adjustment_factor, nan=1.0, posinf=1.0, neginf=1.0)
        # Store it
        store_as_netcdf(
            adjustment_factor=np.reshape(adjustment_factor, grid_shape),
            dataset_example=xr.open_dataset(os.path.join(work_dir, "input", "gridded_rainfall.nc")),
            outfile=os.path.join(work_dir, "output", "adjusted_gridded_rainfall.nc"),
        )
        logger.info("Adjusted gridded rainfall stored to a netCDF.")

    # pylint: disable=broad-exception-caught
    except Exception as exception:
        logger.exception(exception, exc_info=True)

    end = time.perf_counter()
    logger.info(f"Finished rain gauge adjustment. {len(obs_names)} gauges were provided.")
    logger.info("Total adjustment workflow took %s minutes", ((end - start) / 60.0))


if __name__ == "__main__":
    main()
