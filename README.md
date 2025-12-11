# pyRainAdjustment
pyRainAdjustment is an open-source Python toolset to downscale and correct gridded rainfall products using rain gauges. This tool interacts with Delft-FEWS and takes as input a netCDF of gridded rainfall (or other meteorological) product and one or multiple netCDF(s) containing the rain gauge information. It returns a field of correction factors as a netCDF that can be read by Delft-FEWS.

pyRainAdjustment offers the following tools:
- Gridded rainfall downscaling using a climatology-based downscaling procedure.
- Gridded rainfall adjustment in a hindcasting mode. For this, the following methods can be used: mean field bias adjustments, additive, multiplicative and mixed error model correction, and kriging with external drift adjustment.
- Gridded rainfall adjustment in a forecasting mode through quantile mapping.

<img width="1329" height="594" alt="image" src="https://github.com/user-attachments/assets/cac343f2-d8ab-4847-8158-3d8a2cb83aa1" />

This github page provides the source code of pyRainAdjustment, which can be used as a stand-alone module and as a module in your own Delft-FEWS configuration. The description of pyRainAdjustment, what you can do with it and how to install and configure it, are described in the documentation of this github page: [/docs/Introduction.md](https://github.com/Deltares-research/pyRainAdjustment/tree/main/docs/Introduction.md). See ["How to use pyRainAdjustment"](https://github.com/Deltares-research/pyRainAdjustment?tab=readme-ov-file#how-to-use-pyrainadjustment) for more information and to assist you in navigating to the right pages on this github. In addition, the Delft-FEWS public wiki page has [a specific page for pyRainAdjustment](https://publicwiki.deltares.nl/pages/viewpage.action?pageId=356778825&spaceKey=FEWSDOC&title=pyRainAdjustment) where the interaction of this Python tool with Delft-FEWS is elaborated on. On top of that, the Delft-FEWS wiki page can assists you in configuring your Delft-FEWS system and ensuring that pyRainAdjustment works seamlessly in your environment.

## Installation

Make sure you have a Python package manager, such as mamba, micromamba or miniforge. 

Then, in a command prompt or shell, run:

`git clone https://github.com/Deltares-research/pyRainAdjustment.git`

`cd pyRainAdjustment`

`mamba create -n rainadjustment python=3.12 poetry`

`mamba activate rainadjustment`

`poetry install`

## How to use pyRainAdjustment
pyRainAdjustment can be called, preferrably through Delft-FEWS, by calling Python with the following information:
```
cd %REGION_HOME%/Modules/pyrainadjustment 
%REGION_HOME%/Modules/python/python.exe main.py --xml_config input/adjustment_settings.xml --requested_functionality [functionality]
```
with in the `adjustment_settings.xml` the additional properties and settings that are described in [/config/README.md](https://github.com/Deltares-research/pyRainAdjustment/tree/main/config/README.md). The `[functionality]` can be one of the following options: `adjustment`, `downscaling`, `qq_mapping`. 

A general introduction to pyRainAdjustment is provided in [/docs/Introduction.md](https://github.com/Deltares-research/pyRainAdjustment/tree/main/docs/Introduction.md). A background on the various options and methods in pyRainAdjustment is provided under: [/docs/Downscaling.md](https://github.com/Deltares-research/pyRainAdjustment/tree/main/docs/Downscaling.md) for the downscaling method, [/docs/Hindcasting_adjustments.md](https://github.com/Deltares-research/pyRainAdjustment/tree/main/docs/Hindcasting_adjustments.md) for the adjustment methods in hindcasting mode and [/docs/Forecasting_adjustment.md](https://github.com/Deltares-research/pyRainAdjustment/tree/main/docs/Forecasting_adjustment.md) for the adjustment methods in forecasting mode. Examples of how to configure these options in Delft-FEWS are provided in the folder `config` and this is further explained in [/config/README.md](https://github.com/Deltares-research/pyRainAdjustment/tree/main/config/README.md).

## License

TO DO

## Contributions

Contributions to pyRainAdjustment are much appreciated. However, to ensure continuity and a stable code base, we ask you to do the following when you plan to contribute with changes, new config or new models:

*(1) Fork the repository*

*(2) Create a new branch*

All contributions should be made in a new branch under your forked repository. Working on the master branch is reserved for Core Contributors only. Core Contributors are developers that actively work and maintain the repository. They are the only ones who accept pull requests and push commits directly to the pysteps repository.For more information on how to create and work with branches, see [Branches in a Nutshell](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) in the Git documentation.

*(3) Create a pull request based on the changes in your branch*

The pull requests are checked by the main contributors and merged with the main repository once accepted.

### Commit message convention

For commits, we follow the [conventional commits specification](https://www.conventionalcommits.org/en) for our commit messages:

- `fix`: bug fixes, e.g. fix crash due to deprecated method.
- `feat`: new features, e.g. add new method to the module.
- `refactor`: code refactor, e.g. migrate from class components to hooks.
- `docs`: changes into documentation, e.g. add usage example for the module..
- `test`: adding or updating tests, eg add integration tests using detox.
- `chore`: tooling changes, e.g. change CI config.

Make sure to use squash and merge upon merging pull requests.

### Python style conventions

For predefined Python style conventions of this package, see [/docs/Conventions.md](https://github.com/Deltares-research/pyRainAdjustment/tree/main/docs/Conventions.md).
