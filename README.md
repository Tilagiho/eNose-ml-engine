# eNose-ml-engine

## Dependencies
The training scripts use fastai, which is only supported on linux. All other scripts still work on other systems. The easiest way to use the scripts provided in this repository is to setup a conda environment:
1) Clone the repository using [git](https://git-scm.com/) to the desired location.
2) [Install Miniconda on your machine](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda).
3) Setup the environment (eNoseEnv): <br>
```conda create -n eNoseEnv python=3.7```
4) Activate environment: <br>
```conda activate eNoseEnv```
5) Install dependencies:
- Basic dependencies:
    ```pip install  scikit-learn natsort```
- On Linux: fastai
    ```pip install  fastai```
- otherwise:
    ```pip install  torch```

6) (Optional) Test the environment:
    1) cd to directory of repository
    2) start python console: ```python```
    3) Test loading a dataset (replace <pathToDataset> with real path):
    ```
    from funcdataset import *
    dataset = FuncDataset(<pathToDataset>)
    ```
    
    
