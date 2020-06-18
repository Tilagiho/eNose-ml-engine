# eNose-ml-engine

## Dependencies
The easiest way to use the scripts provided in this repository is to setup a conda environment:
1) [Install Miniconda on your machine](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda).
2) Setup the environment (eNoseEnv): <br>
```conda create -n eNoseEnv python=3.7```
3) Activate environment: <br>
```conda activate eNoseEnv```
4) Install dependencies:
```pip install fastai scikit-learn natsort```
5) (Optional) Test the environment:
    1) cd to directory of repository
    2) start python console: ```python```
    3) Test loading a dataset (replace <pathToDataset> with real path):
    ```
    from funcdataset import *
    dataset = FuncDataset(<pathToDataset>)
    ```
    
    
