# pyRainAdjustment Python style conventions and assumptions

For contributions to the Python code in this folder, we try to stick to the following conventions:

## Dimension names:
- lon
- lat
- time
- Main variable of gridded precipitation: default is `P`, but if the dataset has only variable, this is flexible in the code. 

## Variable names:
- We use snake case for variable names, e.g. `precip_gridded` for the gridded precipitation, `precip_gauges` for the rain gauge observations and `gridded_rainfall_forecast` for the gridded rainfall forecast.
- Try to make the variable names intuitive, so `precip` or instead of `p`, etc.

## Path names:
- For directory names, we use the variable extension `_dir` in the variable name. For instance, `work_dir`.
- For filenames, we use the variable extension `_file` in the variable name. For instance, `gridded_file`.

## Climatology:
In the current setup of the tooling, we expect the climatology file to be a netCDF file with the climatology stored per month. 

## Python environment
The installation of python packages in the environment and `pyproject.toml` happens through Poetry, which ensures that all package version get stored in the `poetry.lock` file. 

- At the first use, packages can be installed through: `poetry install` (see the [README](https://github.com/Deltares-research/pyRainAdjustment/blob/main/README.md)).
- As a developer, you can add packages to the `poetry.lock` file with: `poetry add [package_name]`.
- After you have pulled a newer version of the code to your local repository, you can update your environment with: `poetry update`. 

## Other scripting rules
- Maximum line length is set to 100. This can easily be adjusted automatically using the Black package. 

## Wishlist of thing we would like to add for clean and stable coding:
- Tests
- Pre-commit checks
- Typing
