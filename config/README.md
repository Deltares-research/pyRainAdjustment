# config folder

This folder contains example Delft-FEWS configuration (in a template format) that can be used to interact with the pyRainAdjustment Python toolbox. Ensure that your Delft-FEWS environment uses a Python distribution, which can be specified in the `global.properties` file, for instance as:

`PYTHON_EXE = %REGION_HOME%/Modules/Python311/python.exe`

## Configuration options
Besides the example configuration files in the sub folders of this folder, pyRainAdjustment can be fully configured with a list of options that are displayed in the talbe below. These configuration options are passed to pyRainAdjustment through an xml file that is exported by Delft-FEWS. These properties can be configured through a combination of the ([ModuleConfigFiles](https://github.com/Deltares-research/pyRainAdjustment/tree/main/config/ModuleConfigFiles)) and the ([WorkflowDescriptors](https://github.com/Deltares-research/pyRainAdjustment/tree/main/config/WorkflowDescriptors)).

| Configuration Parameter       | Functionality Applied To              | Explanation |
|------------------------------|----------------------------------------|-------------|
| `clim_filepath`              | Downscaling                            | The file path to where the climatology file(s) is (are) stored. |
| `downscaling_factor`         | Downscaling                            | The factor with which the original grid resolution is downscaled. “2” would indicate that the resolution would become two times higher. This value is always an integer. |
| `adjustment_method`          | Adjustment                             | The used adjustment method (provided as a string). This can be: MFB, additive, multiplicative, mixed, KED and quantile_mapping. |
| `statistical_function`       | Adjustment                             | The statistical operator that is used to find the best matching cell/value for the provided gauge location (provided as a string). Options are: mean, median, best, min and max. |
| `threshold`                  | Adjustment                             | Threshold (float; in the unit of the data, e.g. mm or mm h⁻¹), which assures that adjustment only takes place when both the gauge and corresponding grid cell/value are above the threshold. |
| `max_change_factor`          | Adjustment                             | Maximum adjustment factor (float) that can be applied to increase/decrease the grid cell values. Everything above this value will be capped to the maximum adjustment value. |
| `nearest_cells_to_use`       | Adjustment                             | The size of the search grid box to find the corresponding grid cells for a given rain gauge location (integer). |
| `min_gauges`                 | Adjustment                             | The minimum number of gauges required to perform the adjustment procedure (integer). |
| `kriging_n_gauges`           | KED adjustment and kriging interpolation | The minimum number of gauges required to perform either the KED adjustment or kriging interpolation (integer). |
| `interpolation_method`       | Adjustment                             | The interpolation method used to map the adjustment factors back onto the original grid (string). |
| `smooth_edge_values_range`   | Adjustment                             | Optional distance in grid cells (int) from the edge of the domain that will be used smooth the spatial adjustment factors from the center of the domain to the MFB factors on the edge of the domain, which is useful when gauge are only present on a subset of the domain. |
| `variogram_model`            | KED adjustment                         | The variogram model used for the KED adjustment (string). Options are: standard and auto_derive. |
| `derive_qmapping_factors`    | Quantile mapping                       | Setting (Boolean) to either derive the quantile mapping factors (when set to True) or to apply the quantile mapping correction (when set to False). |
| `qmapping_month`              | Quantile mapping                       | If specified (integer value), pyRainAdjustment will only derive the quantile mapping factors for the indicated month. If set to None (the default), it will derive the factors for all months in the data. |
| `leadtime_specific_factors`  | Quantile mapping                       | Setting to derive lead-time specific correction factors (when True) or one factor for all leadtimes (when False). Defaults to False. |
| `qq_filepath`                | Quantile mapping                       | The file path where the quantile mapping factors should be / are stored. |
| `gridded_reference_product`  | Quantile mapping                       | Boolean setting to indicate whether the reference rainfall for the quantile mapping procedure is a gridded product (True) or contains point observations from rain gauges (False). |
