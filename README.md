# eNose-ml-engine

## Dependencies
The easiest way to use the scripts provided in this repository is to setup a conda environment:
1) [Install Anaconda on your machine](https://docs.anaconda.com/anaconda/install/).
2) Setup the environment (eNoseMLEnv): <br>
```conda create -n eNoseMLEnv python=3.7```
3) Activate environment: <br>
```conda activate eNoseMLEnv```
4) Install dependencies:
    - pytorch & fastai: <br>
    ```conda install -c pytorch -c fastai fastai```
    - scikit-learn: <br>
    ```conda install scikit-learn```
5) (Optional) Test environment:
    1) cd to directory of repository
    2) start python console: ```python```
    3) Test loading a dataset (replace <pathToDataset> with real path):
    ```
    import funcdataset
    dataset = funcdataset.FuncDataset(<pathToDataset>)
    ```
    
    
