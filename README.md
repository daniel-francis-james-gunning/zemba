# ZEMBA
A simple energy balance model of the climate system written in Python.

## What is it?
ZEMBA is a **Z**onally-Averaged **E**nergy and **M**oisture **BA**lance Climate Model (**ZEMBA**) written in Python. Put simply, the model estimates the zonal (east-west) average of surface temperatur and precipitation for latitude circles.

## What is its purpose?
ZEMBA has been created to study the response of the climate system to Milankovitch cycles- which describe periodic changes in the configuration of the Earth's orbit around the Sun that alters the distribution of solar radiaition the Earth recives at the top of its atmosphere. These Milankovitch cycles are widely considered to be important for driving the transition between cold 'glacial' period and warmer 'interglacials' period that has occured frequently during the last 2.5 million years of Earth's history.

## Instructions for use of Github repository

There are two options for use of ZEMBA

(1) Download the repository as a ZIP file using the green 'code' button at the top right. 

(2) Install [Git](https://git-scm.com/) (if not already installed) and clone to the repository to your computer. Go to your command line and change your current working directory to where you want to ZEMBA code to be stored. Do this using the 'cd' command (or 'pushd' on Windows) and then clone the repository using the URL.

```
cd your_path_to_the_zemba_code
git clone https://github.com/l975421700/Finse_data_analysis
```

## Setting up the conda environment

Running ZEMBA requires some python packages, most importantly [NUMBA](https://numba.readthedocs.io/en/stable/user/5minguide.html)- a compiler for python NumPy arrays, functions and loops which improves performance. In addition, there are some python packagaes used for plotting and comparing ZEMBA output to state-of-the art climate models and reanalysis products, including [proplot](https://proplot.readthedocs.io/en/latest/index.html) and [xarray](https://docs.xarray.dev/en/stable/index.html)

We recommend creating a new conda environment in the command line (named in this example "zemba_env") with the following packages included:

```
conda create -n zemba_env python numpy numba xarray proplot matplotlib pandas
```

## Running the model

To run an equillibirum simulation of the model for the pre-industrial, you can use either the 'equilirun_template.py' in an integrated development environment (IDE) of your choice or go through the the jupyter notebook file "example_run.ipynb".

More details on ZEMBA can be found in the "example_run.ipynb" jupyter notebook, which goes through an example of running the model and plotting the data....

The model and this Github page are still in development. If anything is unclear or not working, contact me at : daniel.gunning@uib.no

## Pre-Release DOI

DOI: 10.5281/zenodo.10732139
