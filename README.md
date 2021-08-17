# Automatic arrival time picking for seismic inversion with unlabeled data

This code is a Keras implementation of [PhaseNet](https://github.com/wayneweiqiang/PhaseNet), dedicated to *"Automatic arrival time picking for seismic inversion with unlabeled data"*. This version allows to deal with the new dataset having the different size and number of channels, especially, to implement transfer learning using the pretrained model from [NCEDC](https://ncedc.org/) data (Northern California Earthquake Data Center) and deal with small labeled datasets or unlabeled datasets using semi-supervised learning. The robust linear regression methods and SVR (Support Vector Regression) are used to correct labels after pseudo-labelling (see `correct_label` directory) that help improve significantly the quality of pseudo labels.

The model stored in `model/210716-150459` has been trained with 36,864 seismograms from 4 labeled datasets (non-filtered set, filtered sets at 50Hz, 110Hz and 200Hz) using transfer learning with the pretrained model from NCEDC. The data in `dataset/raw/data` is an extraction from the dataset filtered at 50Hz.

The following table sums up the results for our model on training-validation and test datasets, with 27,648 and 9,216 samples respectively:
|                          | Phase | F1-score  |
| ------------------------ | ----- | --------- |
| Training from scratch    |P<br>S | 99.94%<br>82.81% |
| Using transfer learning  |P<br>S | 99.95%<br>83.01% |

## 0. Installing packages
Setting up a virtual environment using Anaconda:
```
conda create --name venv python=3.8
conda activate venv
conda install scikit-learn=0.24 tensorflow=2.5 pandas=1.3 matplotlib=3.4
```

## 1. Data preprocessing
In the terminal command, go to the `dataset` directory:
```
cd dataset
```
- For generating labeled data (for train and test):
```
conda activate venv
python run.py --mode=mode --data_augmentation --data_dir=raw/data --label_list=raw/label_time.csv
```
where `mode` is either `train` (for generating train test), or `test` (for generating test set). The action `--data_augmentation` is used in order to increase the size of train or test set. We can further add the action `--plot_figure` and set a value for the parameter `--plot_rate` in order to verify the data preprocessing results.

- For generating unlabeled data (for prediction):
```
conda activate venv
python run.py --mode=pred --data_dir=raw/data
```
Notes:

- The csv file `dataset/raw/label_time.csv` should be contained 2 columns: `itp`, `its`. Its arrival times should be sorted in ascending alphabetical order of the file name in the `dataset/raw/data` directory.
- The data files in `dataset/raw/data` should be contained 2 columns (the one with the time and the other with the amplitude of the signal).
- If you want to preprocess a new raw data, you can modify some parameters of the `Config()` class in `dataset/data_preprocessing.py`.

## 2. Training
Now, go back to the main directory:
```
cd ..
```
### Training from scratch:

- Training by splitting data into train and valid set with 0.8 of training and 0.2 of validation: 
```
conda activate venv
python train_model.py --valid=0.2 --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --optimizer=adam
```
- Training on the whole dataset:
```
conda activate venv
python train_model.py --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --optimizer=adam
```
### Transfer learning:

- Initializing weights from a pretrained model. By default, the process will load weights from all layers of pretrained model and select all layers of new model for fine-tuning: 
```
conda activate venv
python train_model.py --valid=0.2 --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --model_dir=model/pretrained_from_NCEDC
```
- For more options of this process, you can see the function `ind_layers()` in `train_model.py` along with the summary file of pretrained model in order to select the index of layers for loading and freezing weights when using transfer learning. In this scenario, you can modify the `TO IMPLEMENT` part in this function and set the action `--tune_transfer_learning` as the command below:
```
conda activate venv
python train_model.py --valid=0.2 --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --model_dir=model/pretrained_from_NCEDC --tune_transfer_learning
```
Notes:

- While using transfer learning, make sure that you load weights into layers of the new model that have the same dimension with those of the pretrained model.
- For training with a new data, you can modify some parameters of the `Config()` class in `data_reader.py`.

## 3. Test

- Testing the performance of the model on the test set:
```
conda activate venv
python prediction_model.py --test --model_dir=model/210716-150459 --data_dir=dataset/test/data --data_list=dataset/test/fname.csv --batch_size=100 --save_result --plot_figure
```
## 4. Prediction

- Predicting arrival times for the prediction set using a trained model:
```
conda activate venv
python prediction_model.py --model_dir=model/210716-150459 --data_dir=dataset/pred/data --data_list=dataset/pred/fname.csv --batch_size=100 --save_result --plot_figure
```
## 5. Correcting picks (optional)

This part allows to improve the quality of pseudo labels for semi-supervised learning. In order to correct picks after predicting, go to the [`correct_label`](https://github.com/nghitruyen/PhaseNet_keras_version/tree/main/correct_label) directory.
