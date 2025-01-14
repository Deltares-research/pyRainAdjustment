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
import os

import numpy as np
import xarray as xr
import wradlib as wrl


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
    obs_values = None

    # Open the files
    gauges_files = os.listdir(gauge_folder)
    for gauge_file in gauges_files:
        obs_values_ = []
        if gauge_file.endswith("_Gauges.nc"):
            ds = xr.open_dataset(os.path.join(gauge_folder, gauge_file))

            for t in range(ds.P.shape[0]):
                obs_values_.append(ds.P[t].values)

            # Per station, store the location, station number and the rainfall
            # value
            precip_gauges = ds.P[-1]
            # Check the dimensions of the DataArray and if needed, adjust them
            precip_gauges = check_dimensions(precip=precip_gauges)

            for station_index in range(precip_gauges.shape[0]):
                # Get the station information per station
                station_lat = precip_gauges[station_index].y.values
                station_lon = precip_gauges[station_index].x.values
                station_name = precip_gauges[station_index].station_id.values

                # Append all the values to the output list
                obs_coords.append([float(station_lon), float(station_lat)])
                obs_names.append(str(station_name.tobytes().decode("utf-8").rstrip("\x00")))

            if obs_values is not None:
                obs_values = np.concatenate((obs_values, obs_values_), axis=1)
            else:
                obs_values = obs_values_.copy()

    return np.array(obs_coords), np.array(obs_names), np.array(obs_values)


def obtain_gridded_rainfall_information(grid_file):
    """
    Parameters
    ----------
    grid_file: str
        The netCDF file containing the gridded rainfall information for
        the last hour.

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
    ds_gridded = xr.open_dataset(grid_file)
    precip_gridded = ds_gridded.P

    # Check the dimensions of the DataArray and if needed, adjust them
    precip_gridded = check_dimensions(precip=precip_gridded)

    # Get the grid information
    grid_lats = precip_gridded.y.values
    grid_lons = precip_gridded.x.values
    grid_coords = wrl.util.gridaspoints(grid_lats, grid_lons)
    grid_shape = len(grid_lats), len(grid_lons)
    grid_values = np.array(
        [precip_gridded[t].values.flatten() for t in range(precip_gridded.shape[0])]
    )

    return grid_coords, grid_values, grid_shape


def store_as_netcdf(adjustment_factor, dataset_example, outfile):
    """
    Saves the output adjustment factors to a new NetCDF file.

    Parameters
    ----------
        adjustment_factor: ndarray
            3D array (time, y ,x) containing the adjustment factors on the
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
                adjustment_factor,
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


def check_dimensions(precip):
    """
    Parameters
    ----------
    precip: xr.DataArray
        DataArray containing the rainfall information (scalar or gridded)
        of which the dimensions need to be checked.

    Returns
    ------
    precip: xr.DataArray
        Same DataArray but with checked, and if needed adjusted, x and y
        dimensions.
    """
    # Check if the dataset contains x and y. If it only contains lat, lon
    # or latitude and longitude, rename.
    if "x" not in precip.coords:
        if "lon" in precip.coords:
            precip = precip.rename({"lon": "x"})
        elif "longitude" in precip.coords:
            precip = precip.rename({"longitude": "x"})
        else:
            raise KeyError(
                "The provided DataArray does not contain the dimensions x, lon or longitude"
            )
    if "y" not in precip.coords:
        if "lat" in precip.coords:
            precip = precip.rename({"lat": "y"})
        elif "latitude" in precip.coords:
            precip = precip.rename({"latitude": "y"})
        else:
            raise KeyError(
                "The provided gridded DataArray does not contain the dimensions y, lat or latitude"
            )

    return precip
