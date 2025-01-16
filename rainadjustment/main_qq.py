import argparse
from datetime import datetime, timedelta
import logging
import os
import time

import numpy as np
import pandas as pd
import wradlib as wrl
import xarray as xr
from xarray import Dataset

from rainadjustment.xml_config_parser import parse_run_xml
import rainadjustment.main as main

def create_yearly_netcdf(ds_dir, ds_file):
    dataset = xr.load_dataset(ds_dir + ds_file)
    unique_years = np.unique(dataset.time.dt.year.values)
    print(unique_years)
    ds_list = []
    for year in unique_years:
        start, end = datetime(year, 1, 1), datetime(year, 12, 31)
        ds = dataset.sel(time=slice(start, end))
        ds = ds.resample(time='1D').sum()
        ds_filename = f'./intermediate/{ds_file.split(".nc")[0]}_y{year}.nc'
        ds = ds.assign({'year': [year]})
        ds_list.append(ds_filename)
        ds.to_netcdf(ds_filename)
    return ds_list


def preprocess_to_yearly_files(dir, identifier):
    '''
    Function to make sure there are no duplicate calendardays in the dataset
    by writing the file to multiple separate years.
    Parameters
    ----------
    dir
    identifier

    Returns
    -------

    '''
    file_list = []
    files_in_dir = os.listdir(dir)
    ## Check if grids are larger than 1 year, if so, make separate yearly files
    for file in files_in_dir:
        if identifier in file:
            yearly_files = create_yearly_netcdf(dir, file)
            print('adding to gridlist')
            file_list += yearly_files
    return file_list


def create_grid_climatology(obs_dir: str,
                            grid_dir: str,
                            file_identifier: str):
    obs = main.obtain_gauge_information(obs_dir)
    obs_locs = obs[0]
    station_x = xr.DataArray(obs_locs[:, 0], dims='stations')
    station_y = xr.DataArray(obs_locs[:, 1], dims='stations')
    station_name = xr.DataArray(obs[1], dims='stations')
    grid_files = preprocess_to_yearly_files(grid_dir, file_identifier)
    grid_list = []
    # Spatial selection to trim storage
    for grid_file in grid_files:
        if file_identifier in grid_file:
            grid_ds = xr.load_dataset(grid_file)
            grid_list.append(grid_ds)
    gridded_matches = xr.concat(grid_list, dim='stations')

    # Temporal selection
    min_year, max_year = datetime(1995, 1, 1), datetime(2025, 12, 31)
    gridded_matches = gridded_matches.sel(time=slice(min_year, max_year))
    new_times = pd.to_datetime(
        {'year': 2024,
         'month': gridded_matches['time'].dt.month.values,
         'day': gridded_matches['time'].dt.day.values,
         'hour': gridded_matches['time'].dt.hour.values}
    )

    # Update the 'time' coordinate with the new times
    gridded_matches = gridded_matches.assign_coords(time=('time', new_times), stations=obs[1])
    gridded_matches['station_x'] = station_x
    gridded_matches['station_y'] = station_y
    gridded_matches['station_name'] = station_name
    gridded_matches.to_netcdf('./intermediate/1_Grid_climatology.nc')
    # res = moving_window_date_selection(gridded_matches, 15, 2024)
    gridded_matches.to_netcdf('./intermediate/2_Grid_climatology.nc')
    return gridded_matches


def create_obs_climatology(obs_dir: str,
                           file_identifier: str):
    obs_files = preprocess_to_yearly_files(obs_dir, file_identifier)
    station_names = main.obtain_gauge_information(obs_dir)[1]
    obs_list = []
    # Spatial selection to trim storage
    for obs_file in obs_files:
        if file_identifier in obs_file:
            obs_ds = xr.load_dataset(obs_file)
            obs_cells = obs_ds.assign({'filename': [obs_file]})
            obs_list.append(obs_cells)
    obs = xr.concat(obs_list, dim='stations')
    obs = obs.assign({'stations': station_names})
    # Temporal selection
    min_year, max_year = datetime(1995, 1, 1), datetime(2025, 12, 31)
    obs = obs.sel(time=slice(min_year, max_year))

    # Update the 'time' coordinate with the new times

    # res = moving_window_date_selection(obs, 15, 2024)
    obs.to_netcdf('./intermediate/Obs_climatology.nc')
    return obs


def moving_window_date_selection(dataset, window_size=15, init_year=2024):
    dataset['time'] = [pd.to_datetime(x) for x in dataset.time.values]
    ref_date = datetime(init_year, 1, 1)
    date_list = []
    for day in range(366):
        ref_date_it = ref_date + timedelta(days=day)
        first_date = ref_date_it - timedelta(days=window_size)
        last_date = ref_date_it + timedelta(days=window_size)
        try:
            if day < window_size:  ## If early in january, so we go back to december
                dec_date, jan_date = first_date, last_date
                temp_ds1 = dataset.sel(time=slice(dec_date + timedelta(days=366), ref_date + timedelta(days=366)))
                temp_ds2 = dataset.sel(time=slice(ref_date, jan_date))
                temp_ds = xr.concat([temp_ds1, temp_ds2], dim='time')
            elif day in range(window_size, 366 - window_size):
                temp_ds = dataset.sel(time=slice(first_date, last_date))
            elif day >= (366 - window_size):  ## If late in december, so we go into january
                dec_date, jan_date = first_date, last_date
                temp_ds1 = dataset.sel(time=slice(dec_date, ref_date + timedelta(days=366)))
                temp_ds2 = dataset.sel(time=slice(ref_date, jan_date - timedelta(days=366)))
                temp_ds = xr.concat([temp_ds1, temp_ds2], dim='time')
        except Exception as e:
            print('No valid data for daynumber: ', day, 'with error: ', e)
            continue
        if len(temp_ds.time.values) > 0:
            temp_ds = temp_ds.assign({'reference_time': [ref_date_it]})
            temp_ds = temp_ds.assign_coords(
                delta_days=("time", (temp_ds.time.values - np.datetime64(ref_date_it)) / np.timedelta64(1, 'D')))
            temp_ds = temp_ds.swap_dims({'time': "delta_days"})

            date_list.append(temp_ds)
    res = xr.concat(date_list, dim='reference_time')
    return res


def create_qmapping_factors(grid_values, obs_values):
    '''
    Past correctie toe op distributie van tijdstap observaties.
    Parameters
    ----------
    grid any hindcast period with any timestep
    obs

    Returns

    -------

    '''
    grid_q = np.quantile(grid_values, q=np.linspace(0, 1, 101))
    obs_q = np.quantile(obs_values, q=np.linspace(0, 1, 101))
    correction_factors = obs_q / grid_q
    return correction_factors, grid_q, obs_q


def apply_qmap_correction(forecast: float, historical_data: np.ndarray[:], cor_factor: np.ndarray[:]):
    '''
    Return quantile mapping factors based on historical forecasts
    Parameters
    ----------
    forecast: value for a specific station
    historical_data: array containing historic forecast distribution for the percentiles
    cor_factor: 1D-array containing correction factors for the percentiles

    Returns
    -------

    '''
    sorted_hist = np.sort(historical_data.flatten())
    hist_percentile = np.interp(forecast, sorted_hist, np.linspace(0, 1, len(sorted_hist)))
    correction = np.interp(hist_percentile, np.linspace(0, 1, 101), cor_factor)
    # corrected_value = forecast * correction
    return correction


def Q_mapping_fitting(config_xml, window_size=15):
    '''

    Returns
    -------

    '''


    ## Take steps to compute correction factors
    grid_hist = create_grid_climatology(config_xml['obs_dir'], config_xml['grid_dir'], config_xml['grid_identifier'])
    obs_hist = create_obs_climatology(config_xml['obs_dir'], config_xml['obs_identifier'])
    obs_values = obs_hist.drop_dims(['filename'])
    obs_accumulated = moving_window_date_selection(obs_hist, 15)
    grid_accumulated = moving_window_date_selection(grid_hist, 15)

    # Testing fix
    obs_accumulated['reference_time'] = grid_accumulated.reference_time.values[:31]
    grid_accumulated = grid_accumulated.sel({'reference_time': obs_accumulated.reference_time.values})
    factors = xr.apply_ufunc(
        create_qmapping_factors,
        grid_accumulated.P,
        obs_accumulated.P,
        input_core_dims=[['reference_time', 'delta_days'], ['reference_time', 'delta_days']],
        output_core_dims=[]
    )
    grid_hist.assign({'correction_factors': (['x', 'y', 'reference_time'], factors)})
    grid_hist.to_netcdf('./intermediate/Grid_climatology.nc')




def Q_mapping_correction_factors(grid_frcst: Dataset, window_size=15):
    climatology = xr.load_dataset('./intermediate/Grid_climatology.nc')
    factors = climatology.correction_factors
    distr = climatology.P

    corrections = xr.apply_ufunc(
        apply_qmap_correction,
        grid_frcst,
        factors,
        distr,
        input_core_dims=[[], [], ['all_calendarday_data']],
        output_core_dimensions=[],
        vectorize=True
    )

    return None

def main_qq():
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
    logging.basicConfig(filename=logfn,
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.info("Arguments parsed:\n %s", global_args)

    # If --xml_config is provided, parse it and prepare to use its values
    config_xml = {}
    if global_args.xml_config:
        try:
            config_xml = parse_run_xml(global_args.xml_config)
        except Exception as exception:
            logger.exception(exception, exc_info=True)
    else:
        try:
            config_xml = parse_run_xml('input/adjustment_settings.xml')
        except Exception as exception:
            logger.exception(exception, exc_info=True)

    # The actual work
    Q_mapping_fitting(config_xml)


if __name__ == '__main__':
    main_qq()