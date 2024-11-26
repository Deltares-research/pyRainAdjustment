# rainadjustment

Python toolset to downscale and correct gridded rainfall products using rain gauges. This tool interacts with Delft-FEWS and takes as input a netCDF of gridded rainfall (or other meteorological) product and one or multiple netCDF(s) containing the rain gauge information. It returns a field of correction factors as a netCDF that can be read by Delft-FEWS.

# Installation
Make sure you have a Python package manager, such as mamba, micromamba or miniforge. 
Then, in a command prompt or shell, run:
$ git clone https://github.com/Deltares-research/pyRainAdjustment.git
$ cd rainadjustment
$ mamba create -n rainadjustment python=3.12 poetry
$ mamba activate rainadjustment
$ poetry install


