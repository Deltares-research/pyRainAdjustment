# pyRainAdjustment

Python toolset to downscale and correct gridded rainfall products using rain gauges. This tool interacts with Delft-FEWS and takes as input a netCDF of gridded rainfall (or other meteorological) product and one or multiple netCDF(s) containing the rain gauge information. It returns a field of correction factors as a netCDF that can be read by Delft-FEWS.

## Installation

Make sure you have a Python package manager, such as mamba, micromamba or miniforge. 

Then, in a command prompt or shell, run:

`git clone https://github.com/Deltares-research/pyRainAdjustment.git`

`cd rainadjustment`

`mamba create -n rainadjustment python=3.12 poetry`

`mamba activate rainadjustment`

`poetry install`

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
