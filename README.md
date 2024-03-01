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
conda create -n zemba_env python numpy numba xarray proplot matplotlib
```

## Running the model

To run an equillibirum simulation of the model for the pre-industrial, you can use either the 'equilirun_template.py' in an integrated development environment (IDE) of your choice or the jupyter notebook file "equilirun_template.ipynb".

Put simply, the model requires a number of inputs to run, including zonal-mean land fractions, land elevation, ice fractions over land, cloud cover over land and ocean, etc. These are all kept in an input file. In this case, the input file is named "input_template.py" stored in the input folder. In "input_template.py", land fractions, land elevations and ice fractions are taken from the ICE-6G-C (Argus et al., 2014; Peltier et al., 2014) and cloud cover is taken from a pre-industrial simulation of NorESM2 (Seland et al., 2020). In the file, you will see the option of using other land fractions (etc). from the Last Glacial Maximum (LGM) around 21 thousand years ago. Alternatively, you could insert land and ice fractions of your choice within the input file.

Once the model is running, the output will be stored in the output folder. In this case, it will be named "template_moist_res5.0.pkl". It contains a nested dictionary of output from the model.

More details on running the model is found in the "equilirun_template.ipynb" notebook, which goes through an example of running the model and plotting the data....

