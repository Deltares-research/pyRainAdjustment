# -*- coding: utf-8 -*-
"""
Functions to read and write netCDF from and for Delft-FEWS.

Available functions:
- obtain_gauge_information
- obtain_gridded_rainfall_information
- store_as_netcdf
- check_dimensions
"""

from datetime import datetime
import logging
import os

import numpy as np
import numpy.typing as npt
import xarray as xr
import wradlib as wrl


def obtain_gauge_information(
    gauge_folder: str, logger: logging.Logger
) -> tuple[npt.NDArray[np.float64], list[str], npt.NDArray[np.float64]]:
    """
    Parameters
    ----------
    gauge_folder: str
        The folder containing all netCDFs with the rain gauge observations.
        The rain gauge observations are expected to contain "_gauges.nc"
        in their name.
    logger: logging instance
        Logger for log messages, passed on from the main.py script.

    Returns
    ------
    obs_coords: ndarray(float)
        List of float containing the latitude and longitude values of the
        gauges as (lat, lon).
    obs_names: list(str)
        List containing the station names.
    obs_values: ndarray(float)
        List containing the station observation values per station.
    """
    # Create the output list
    obs_coords = []
    obs_names = []
    obs_values = None

    # Open the files
    gauges_files = os.listdir(gauge_folder)
    for gauge_file in gauges_files:
        obs_values_ = []
        if gauge_file.endswith("_gauges.nc"):
            ds = xr.open_dataset(os.path.join(gauge_folder, gauge_file))

            for t in range(ds.P.shape[0]):
                obs_values_.append(ds.P[t].values)

            # Per station, store the location, station number and the rainfall
            # value
            if "P" in ds.var():
                precip_gauges = ds.P[-1]
            else:
                var_names = list(ds.var())
                precip_var_name = [element for element in var_names if element not in ["crs", "z"]]
                # Check that there is only one precip_var_name
                if len(precip_var_name) == 1:
                    precip_gauges = ds[precip_var_name]
                else:
                    logger.error(
                        "More than one variable provided in the gauge dataset. Variables found are: %s",
                        precip_var_name,
                    )
                    raise KeyError(
                        "More than one variable provided in the gauge dataset. Variables found are: %s",
                        precip_var_name,
                    )

            # Check the dimensions of the DataArray and if needed, adjust them
            precip_gauges = check_dimensions(precip=precip_gauges, logger=logger)

            for station_index in range(precip_gauges.shape[0]):
                # Get the station information per station
                station_lat = precip_gauges[station_index].lat.values
                station_lon = precip_gauges[station_index].lon.values
                station_name = precip_gauges[station_index].station_id.values

                # Append all the values to the output list
                obs_coords.append([float(station_lon), float(station_lat)])
                obs_names.append(str(station_name.tobytes().decode("utf-8").rstrip("\x00")))

            if obs_values is not None:
                obs_values = np.concatenate((obs_values, obs_values_), axis=1)
            else:
                obs_values = obs_values_.copy()

    return np.array(obs_coords), obs_names, np.array(obs_values)


def obtain_gridded_rainfall_information(
    grid_file: str, logger: logging.Logger
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[int, int]]:
    """
    Parameters
    ----------
    grid_file: str
        The netCDF file containing the gridded rainfall information for
        the last hour.
    logger: logging instance
        Logger for log messages, passed on from the main.py script.

    Returns
    ------
    grid_coords: ndarray(float)
        List of floats containing the latitude and longitude values of the
        gridded rainfall as (lat, lon).
    grid_values: ndarray(float)
        List containing the rainfall values per grid point as a flattened
        array.
    grid_shape: tuple(int, int)
        The shape of the gridded rainfall product before flattening
        (lons, lats)
    """
    ds_gridded = xr.open_dataset(grid_file)

    if "P" in ds_gridded.var():
        precip_gridded = ds_gridded.P
    else:
        var_names = list(ds_gridded.var())
        precip_var_name = [element for element in var_names if element not in ["crs", "z"]]
        # Check that there is only one precip_var_name
        if len(precip_var_name) == 1:
            precip_gridded = ds_gridded[precip_var_name]
        else:
            logger.error(
                "More than one variable provided in the gridded precip dataset. Variables found are: %s",
                precip_var_name,
            )
            raise KeyError(
                "More than one variable provided in the gridded precip dataset. Variables found are: %s",
                precip_var_name,
            )

    # Check the dimensions of the DataArray and if needed, adjust them
    precip_gridded = check_dimensions(precip=precip_gridded, logger=logger)

    # Get the grid information
    grid_lats = precip_gridded.lat.values
    grid_lons = precip_gridded.lon.values
    if len(grid_lats.shape) == 1:
        grid_coords = wrl.util.gridaspoints(grid_lats, grid_lons)
        grid_shape = len(grid_lats), len(grid_lons)
    else:
        grid_coords = np.array(
            [
                [float(grid_lons[i, j]), float(grid_lats[i, j])]
                for i in range(grid_lats.shape[0])
                for j in range(grid_lats.shape[1])
            ]
        )
        grid_shape = grid_lats.shape

    grid_values = np.array(
        [precip_gridded[t].values.flatten() for t in range(precip_gridded.shape[0])]
    )

    return grid_coords, grid_values, grid_shape


def store_as_netcdf(
    gridded_array: npt.NDArray[np.float64],
    dataset_example: xr.Dataset,
    variable_name: str,
    outfile: str,
) -> None:
    """
    Saves the output adjustment factors to a new NetCDF file.

    Parameters
    ----------
        gridded_array: ndarray
            3D array (time, y ,x) containing the adjustment factors on the
            original gridded rainfall product grid.
        dataset_example: xr DataSet
            The original gridded rainfall xr DataSet which will
            form the blue print for the output dataset.
        variable_name: str
            The name of the variable that should be saved.
        outfile: str
            The output file location.

    Returns
    -------
    None
    """
    # Make a dataset out of the array
    output_dataset = xr.Dataset(
        {
            f"{variable_name}": (
                ("time", "y", "x"),
                gridded_array,
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
    output_dataset[f"{variable_name}"].attrs.update(
        {
            "long_name": f"{variable_name}",
            "standard_name": f"{variable_name}",
            "units": "-",
        }
    )

    # Saving reprojected data
    output_dataset.to_netcdf(outfile)
    output_dataset.close()

    return


def check_dimensions(precip: xr.DataArray, logger: logging.Logger) -> xr.DataArray:
    """
    As we only calculate in WGS84 latitude and longitude x- and y-values,
    we first check if these dimensions exist.

    Parameters
    ----------
    precip: xr.DataArray
        DataArray containing the rainfall information (scalar or gridded)
        of which the dimensions need to be checked.
    logger: logging instance
        Logger for log messages, passed on from the main.py script.

    Returns
    ------
    precip: xr.DataArray
        Same DataArray but with checked, and if needed adjusted, x and y
        dimensions.
    """
    # Check if the dataset contains x and y. If it only contains lat, lon
    # or latitude and longitude, rename.
    if "lon" not in precip.coords:
        if "longitude" in precip.coords:
            precip = precip.rename({"longitude": "lon"})
        elif "x" in precip.coords:
            precip = precip.rename({"x": "lon"})
        else:
            logger.error(
                "The provided DataArray does not contain the dimensions x, lon or longitude"
            )
            raise KeyError(
                "The provided DataArray does not contain the dimensions x, lon or longitude"
            )
    if "lat" not in precip.coords:
        if "latitude" in precip.coords:
            precip = precip.rename({"latitude": "lat"})
        elif "y" in precip.coords:
            precip = precip.rename({"y": "lat"})
        else:
            logger.error(
                "The provided gridded DataArray does not contain the dimensions y, lat or latitude"
            )
            raise KeyError(
                "The provided gridded DataArray does not contain the dimensions y, lat or latitude"
            )

    return precip
