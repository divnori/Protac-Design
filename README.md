# Protac-Design
## Description
This repo contains the code behind our workshop paper at the NeurIPS 2022 AI4Science Workshop, [TODO add link to camera-ready version when live]. It is organized into the following notebooks:

* [surrogate_model.ipynb](./surrogate_model.ipynb): Contains the code for processing the raw PROTAC data and training the DC<sub>50</sub> surrogate model. Note that you will need to download the public PROTAC data from [PROTAC-DB](http://cadd.zju.edu.cn/protacdb/downloads) in order to reproduce the results.
* [molecule_metrics.ipynb](./molecule_metrics.ipynb): Contains code for computing metrics on a set of generated molecules. Metrics include percentage predicted active, percentage of duplicate molecules, percentage of molecules regenerated from training set, average number of atoms, chemical diversity, and drug-likeness.
* [binary_label_metrics.py](./binary_label_metrics.py): Contains useful functions for analyzing performance of binary classification models. 

Then there are additional files in the repo:
* [surrogate_model.pkl](./surrogate_model.pkl): Contains the pre-trained surrogate model for DC<sub>50</sub> prediction.
* [features.pkl](./features.pkl): Contains list of features used in surrogate model training; required to reproduce reinforcement learning jobs using protac scoring function.


## Instructions
1. Before running any of the notebooks, you will need to download the PROTAC data from the public [PROTAC-DB](http://cadd.zju.edu.cn/protacdb) database.
2. You will then need to create a conda environment containing the following main packages: `rdkit`, `pandas`, `sklearn`, `scipy`, `ipython`, and `optuna`. See the instructions in the next section for setting this up.
3. Open the notebooks on your favorite platform, and make sure to select the right kernel before executing.

## Environment
To set up the environment for running the notebooks in this repo, you can follow the following set of instructions:
```
conda create -n protacs-env -c conda-forge scikit-learn optuna rdkit
conda activate protacs-env
conda install pandas scipy 
```


## Citation
[TODO add when camera-ready version live]

## Additional data
Additional data, including saved GraphINVENT model states, generated structures, analysis scripts, and training data, are available on Zenodo [here](https://doi.org/10.5281/zenodo.7278277).

## Authors
* Divya Nori
* Rocío Mercado
