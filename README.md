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
    ```
    pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install pandas matplotlib
    ```

6) (Optional) Test the environment:
    1) cd to directory of repository
    2) start python console: ```python```
    3) Test loading a dataset (replace <pathToDataset> with real path):
    ```
    from funcdataset import *
    dataset = FuncDataset(<pathToDataset>)
    ```
    
## Analysing & Training datasets
A dataset measured and annotated with the eNoseAnnotator can be analysed and used for training models in the eNose-ml-engine.
The FuncDatasetClass loads and prepares the dataset. By default, it expects a filepath to a directory containing two directories: "train" containing the measurements of the train set and "valid" containing the measurements of the validation set.
If you only want to analyse the dataset adjust the path to the dataset and configure the settings at the end of "funcdataset.py" and run the script in an interactive python console. 
Afterwards you can call various visual analysation functions:
```
# plot 2-dimensional pca analysis & linear discrimant analysis of the dataset
dataset.plot2DAnalysis()

# plot the values of func 0 against all other functionalisations
dataset.plot_func_relationships(0)

# plot the values of func 0 against all other functionalisations,
# but only use the classes in the list
dataset.plot_func_relationships(0, ["Aceton", "Ethanol", "Isopropanol"])

# plot the func correlation matrix
dataset.plot_correlation_matrix()

# the correlation matrix can be filtered for specific classes as well
dataset.plot_correlation_matrix(["Ammoniak"])
```

If you want to train a feedforward model configure the settings at the beginning of "train.py" and run the script. The class TrainAnalyser can be used afterwards to plot the developements of the different metrics during training.

For training support vector machines use the script "train_svm.py".

You can use the class MultiLabelClassificationInterpretion after the training to analyse the results of feedforward models and svms. 
