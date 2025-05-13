import argparse
import copy
from datetime import datetime, timedelta
import logging
import os
import time
import glob

import numpy as np
import pandas as pd
import wradlib as wrl
import xarray as xr
from xarray import Dataset

from rainadjustment.utils.xml_config_parser import parse_run_xml
from utils.io import obtain_gauge_information

def create_yearly_netcdf(filepath, config, check_na_dim = None, do_resample = False, min_year = 2012):
    dataset = xr.load_dataset(filepath)
    if 'tp' in dataset.data_vars:
        dataset = dataset.rename({'tp': 'P'})
    if check_na_dim != None:
        dataset = dataset.dropna(dim=check_na_dim, how='all', subset='P')
    unique_years = np.unique(dataset.time.dt.year.values)
    print(unique_years)
    unique_years = unique_years[unique_years>= min_year]
    ds_list = []
    for x, year in enumerate(unique_years):
        start, end = datetime(year, 1, 1), datetime(year, 12, 31)
        ds = dataset.sel(time=slice(start, end))
        if do_resample:
            ds['P'] = ds.P.resample(step='3h').sum() ## Need to find uniform dimension to resample on frcst/obs
        ds_filename = f'./{filepath.split(".nc")[0]}_y{year}.nc'
        ds_list.append(ds_filename)
        ds.to_netcdf(ds_filename)
    return ds_list

def preprocess_ECMWF_grib(grib_ds):
    '''
    Function to be applied in the xr.open_mfdataset call to preprocess gribs
    Returns
    -------

    '''
    ds = grib_ds.diff(dim = 'step')
    return ds


def create_grid_obs_pairs(grid_dir: str,
                          obs_dir: str,
                          grid_identifier: str):
    obs = obtain_gauge_information(obs_dir)
    obs_locs = obs[0]
    grid_list = []
    grid_files = os.listdir(grid_dir)
    for grid_file in grid_files:
        if grid_identifier in grid_file:
            grid_ds = xr.load_dataset(grid_dir + grid_file)
            ## Remap years
            print(grid_ds.time.dt.dayofyear.values)
            ## Ugly selecting, takes both nearest indices, does not consider multi-index space
            for obs_lat, obs_lon in obs_locs:
                obs_cells = grid_ds.sel({'x': obs_lat, 'y': obs_lon}, method= 'nearest')
                print(obs_cells)
                grid_list.append(obs_cells)
    return obs_locs, grid_list

def deaccumulate(dataset, variable_name, deacc_dimension_name = 'time'):
    dataset[variable_name] = dataset[variable_name].diff(dim = deacc_dimension_name)
    return dataset

def aggregate_to_timestep(dataset, timestep_hours, do_accumulate = True):
    if do_accumulate:
        dataset = dataset.resample(step = f'{timestep_hours}h').sum()
    else:
        dataset = dataset.resample(step = f'{timestep_hours}h').mean()
    return dataset

def attach_valid_time(dataset, t_name, dt_name):
    data_valid_time = dataset[t_name] + dataset[dt_name]
    dataset = dataset.assign_coords(valid_time=(data_valid_time))
    return dataset

def create_grid_climatology(obs_dir: str,
                            grid_dir: str,
                            file_name: str,
                             obs_identifier: str,
                            config: dict = {}):
    obs = obtain_gauge_information(obs_dir, obs_identifier)
    obs_locs = obs[0]
    station_x = xr.DataArray(obs_locs[:, 0], dims='stations')
    station_y = xr.DataArray(obs_locs[:, 1], dims='stations')
    station_name = xr.DataArray(obs[1], dims='stations')
    grid_files = create_yearly_netcdf(grid_dir+file_name, config, check_na_dim=None, do_resample=False)
    # Spatial selection to trim storage
    grid_list = []
    for grid_file in grid_files:

        grid_ds = xr.load_dataset(grid_file)
        loc_list = []
        for x in range(len(obs_locs)):
            obs_cells = grid_ds.sel({'longitude': obs_locs[x][0], 'latitude': obs_locs[x][1]}, method='nearest')
            obs_cells = obs_cells.assign_coords(stations = (obs[1][x]))
            loc_list.append(obs_cells)
        matched_cells = xr.concat(loc_list, dim = 'stations')
        grid_list.append(matched_cells)


    gridded_matches = xr.concat(grid_list, dim='time')
    gridded_matches['station_x'] = station_x
    gridded_matches['station_y'] = station_y
    gridded_matches['station_name'] = station_name
    gridded_matches = gridded_matches.assign_coords(stations=obs[1])
    gridded_matches = gridded_matches.diff(dim = 'step')
    gridded_matches['P'] = gridded_matches.P * 1000
    gridded_matches['P'] = xr.where(gridded_matches.P > 0.02, gridded_matches.P, 0) ## Trim for 0.02mm threshold
    gridded_matches.to_netcdf('./intermediate/Grid_climatology_2012_2024.nc')
    return gridded_matches


def create_obs_climatology(obs_dir: str,
                           file_identifier: str):
    obs = xr.load_dataset(obs_dir + file_identifier)
    obs = obs.assign_coords(stations = obs.station_id)
    print(obs)
    obs = obs.dropna('stations', how = 'all', subset = 'P')
    obs['stations'] = [x.decode("utf-8") for x in obs['stations'].values]
    obs.to_netcdf('./intermediate/Obs_climatology_2012_2024.nc')
    return obs

## No longer used
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
                delta_t=("time", np.linspace(-window_size*8, window_size*8, 2*8*window_size + 1)))
            temp_ds = temp_ds.swap_dims({'time': "delta_t"})
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
    grid_q = np.nanquantile(grid_values.flatten(), q=np.linspace(0, 1, 201))
    obs_q = np.nanquantile(obs_values, q=np.linspace(0, 1, 201))
    correction_factors = obs_q / grid_q
    return np.array([correction_factors, grid_q, obs_q])


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
    correction = np.interp(forecast, historical_data, cor_factor)
    apply_corr = np.where(correction > 1000, 1000, correction)
    # correction = np.interp(hist_percentile, np.linspace(0, 1, len(historical_data)), cor_factor)
    # corrected_value = forecast * correction
    return np.array([correction, forecast, forecast*apply_corr])


def Q_mapping_gridref(init_month: int, input_file = '.nc'):

    if input_file == '.grib':
        next_month = np.max((1,(init_month + 1 )%13))
        files = glob.glob(f'./input/ECMWF/downloaded*_{str(init_month).zfill(2)}.grib')
        files += glob.glob(f'./input/ECMWF/downloaded*_{str(next_month).zfill(2)}.grib')
        print(files)
        grid_hist=xr.open_mfdataset(files, preprocess= preprocess_ECMWF_grib)
        grid_hist.to_netcdf('./input/ECMWF/complete_grib2.nc')

    grid_hist = xr.load_dataset('./input/ECMWF/complete_grib2.nc')


    grid_hist['longitude'] = np.round(grid_hist.longitude.values, decimals=2)
    grid_hist['latitude'] = np.round(grid_hist.latitude.values, decimals=2)
    select_timestamps = grid_hist.time.dt.month.values == init_month
    grid_hist = grid_hist.isel({'time': select_timestamps})
    grid_hist = grid_hist.sel({'time' : slice('2013-01-01', '2021-01-01')}) ## Fixed lim for hist period
    grid_hist['tp'] = grid_hist.tp.diff(dim = 'step')
    print(grid_hist.valid_time.values)

    ref_hist = xr.load_dataset('./input/ERA5_LAND_MDBA_2013_2020/data_0.nc')
    ref_hist = ref_hist.rename({'valid_time': 'time'})
    print(ref_hist.time.values)
    ref_hist = ref_hist.sel({'time': grid_hist.valid_time.values})
    ref_hist['longitude'] = np.round(ref_hist.longitude.values, decimals=2)
    ref_hist['latitude'] = np.round(ref_hist.latitude.values, decimals=2)

    # Find common space in dimension, trim datasets
    for dim in ['longitude', 'latitude']:
        print(grid_hist[dim].values.min(), ref_hist[dim].values.min())
        min = np.max([grid_hist[dim].values.min(), ref_hist[dim].values.min()])
        max = np.min([grid_hist[dim].values.max(), ref_hist[dim].values.max()])
        print(min, max)
        subset = slice(min, max)
        if dim == 'latitude':
            subset = slice(max, min)
        grid_hist = grid_hist.sel({dim: subset})
        ref_hist = ref_hist.sel({dim: subset})

    q_list = []
    for step_it, leadtime in enumerate(grid_hist.step.values):
        print(step_it)
        grid_at_step = grid_hist.isel({'step': step_it})
        ref_hist_step = ref_hist.sel({'time': grid_hist.valid_time.values})
        factors = xr.apply_ufunc(
            create_qmapping_factors,
            grid_at_step.tp,
            ref_hist_step.tp,
            input_core_dims=[['time'], ['time']],
            output_core_dims=[['variable', 'percentile']],
            vectorize=True,
            join = 'outer'
        )
        q_list.append(factors)

    q_array = xr.concat(q_list, dim = 'step')
    q_array = q_array.assign_coords(percentile = np.linspace(0,1,201))
    q_array = q_array.to_dataset(dim = 'variable')
    q_array = q_array.rename({0: 'correction_factors', 1: 'forecast_P_at_q', 2: 'obs_P_at_q'})

    print(q_array.correction_factors.values)
    q_array.to_netcdf('./intermediate/grid_quantile_correction.nc')
    return q_array


def Q_mapping_fitting(config_xml, init_month: int):
    '''

    Returns
    -------

    '''


    ## Take steps to compute correction factors
    # grid_hist = create_grid_climatology(config_xml['obs_dir'], config_xml['grid_dir'], config_xml['grid_identifier'], config_xml['obs_identifier'])
    grid_hist = xr.load_dataset('./intermediate/Grid_climatology_2012_2024.nc')
    # grid_hist['P'] = grid_hist.P * 1000
    obs_hist = xr.load_dataset('./intermediate/Obs_climatology_2012_2024.nc')
    min_non_zero = copy.deepcopy(obs_hist.P.values)
    min_non_zero[min_non_zero == 0] = np.nan
    min_non_zero = np.nanmin(min_non_zero, axis=0)
    print(min_non_zero)
    obs_hist = obs_hist.dropna('stations', how = 'all', subset = 'P')
    grid_hist = grid_hist.sel({'stations': obs_hist.stations.values})
    # obs_hist = create_obs_climatology(config_xml['obs_dir'], config_xml['obs_identifier'])

    select_timestamps = grid_hist.time.dt.month.values == init_month

    grid_collection = grid_hist.isel({'time': select_timestamps})
    q_list = []
    for step_it, leadtime in enumerate(grid_collection.step.values):
        print(step_it)
        grid_at_step = grid_collection.isel({'step': step_it})
        select_times = grid_at_step.valid_time.values.flatten()
        select_times_trimmed = select_times[select_times<=obs_hist.time.values.max()] ## Trim end of timestamps if obs not available
        if len(select_times) - len(select_times_trimmed) > 91:
            raise ValueError('Timestamp Trimming Error: Too many observations not available')
        obs_values = obs_hist.sel({'time': select_times_trimmed})
        factors = xr.apply_ufunc(
            create_qmapping_factors,
            grid_at_step.P,
            obs_values.P,
            input_core_dims=[['time'], ['time']],
            output_core_dims=[['variable', 'percentile']],
            vectorize=True,
            join = 'outer',
        )
        q_list.append(factors)

    q_array = xr.concat(q_list, dim = 'step')
    q_array = q_array.assign_coords(percentile = np.linspace(0,1,201))
    q_array = q_array.to_dataset(dim = 'variable')
    q_array = q_array.rename({0: 'correction_factors', 1: 'forecast_P_at_q', 2: 'obs_P_at_q'})

    print(q_array)
    q_array.to_netcdf('./intermediate/Quantile_climatology.nc')
    return q_array

def Q_mapping_correction_factors(grid_frcst: Dataset, grid_clim_path):
    climatology = xr.load_dataset(grid_clim_path)
    factors = climatology.correction_factors
    distr = climatology.forecast_P_at_q

    # grid_frcst = grid_frcst.sel({'stations': climatology.stations.values})

    print(grid_frcst.dims)
    print(climatology.dims)

    corrections = xr.apply_ufunc(
        apply_qmap_correction,
        grid_frcst.tp,
        distr,
        factors,
        input_core_dims=[['time'], ['percentile'],['percentile']],
        output_core_dims = [['variable', 'time']],
        vectorize=True
    )
    print(corrections)

    return corrections

def main_qq(year, end_year, month):
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

    forecast = xr.load_dataset('./input/ECMWF/complete_grib2.nc')
    forecast = forecast.sel({'time': forecast.time.values>np.datetime64(f'{str(year)}-01-01')})
    forecast = forecast.sel({'time': forecast.time.values<np.datetime64(f'{str(end_year + 1)}-01-01')})
    forecast = forecast.isel({'time': forecast.time.dt.month == int(month)}) ## Select March

    forecast['longitude'] = np.round(forecast.longitude.values, decimals=2)
    forecast['latitude'] = np.round(forecast.latitude.values, decimals=2)
    forecast['tp'] = forecast.tp.diff(dim = 'step')
    forecast.tp.values[forecast.tp.values<1e-06] = np.nan

    print(forecast)

    corrections = Q_mapping_correction_factors(forecast, './intermediate/grid_quantile_correction.nc')
    print(corrections)
    corrections = corrections.to_dataset(dim = 'variable')
    print(corrections)
    corrections = corrections.rename({0: 'correction', 1: 'P_forecast', 2: 'P_corrected'})
    print(corrections)
    corrections.to_netcdf(f'./output/gridded_corrections_{year}-{end_year}_{month}.nc')
    return corrections

if __name__ == '__main__':
    Q_mapping_gridref(input_file = '.grib', init_month=3)
    main_qq(2019, 2020, '03')