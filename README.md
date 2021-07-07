This code is a Keras implementation of [PhaseNet](https://github.com/wayneweiqiang/PhaseNet), dedicated to "Automatic arrival time picking for seismic inversion". This version allows to deal with the new dataset having the different size and number of channels, especially, to implement transfer learning using the pretrained model from the [NCEDC](https://ncedc.org/) data (Northern California Earthquake Data Center). The model stored in `model/210706-105857` has been trained with 27,648 seismograms after expanding 3 times the size of dataset by using data augmentation. The data in `dataset/raw/data` is an extract from 9,216 seismograms of the dataset.

## 0. Installing packages
### Using anaconda (recommend):
```
conda create --name venv python=3.8
conda activate venv
conda install tensorflow=2.5 matplotlib scipy pandas
```
### Using virtualenv:
```
pip install virtualenv
virtualenv .venv -p python3.8
source .venv/bin/activate
pip install -r requirements.txt
```
## 1. Data preprocessing
Go to the `dataset` directory, the raw data is stored in `raw`.

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

- The csv file `dataset/raw/label_time.csv` must contain 2 columns: `itp`, `its`. Its arrival times must be sorted in ascending alphabetical order of the file name in the `dataset/raw/data` directory.
- The data files in `dataset/raw/data` must contain 2 columns (the one with the time and the other with the amplitude of the signal).
## 2. Training
Now, go back to the main directory.
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

- Initializing weights from a pretrained model. By default, the process will load weights from all layers of pretrained model and fine-tune weights in all of these layers: 
```
conda activate venv
python train_model.py --valid=0.2 --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --model_dir=model/pretrained_from_NCEDC
```
- For more options of this process, you can see the function `ind_layers()` in `train_model.py` as well as the summary file of pretrained model in order to chose the index of layers for loading and freezing weights when using transfer learning. In this scenario, you need to modify the `TO IMPLEMENT` part in this function and set the action `--tune_transfer_learning` as the command below:
```
conda activate venv
python train_model.py --valid=0.2 --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --model_dir=model/pretrained_from_NCEDC --tune_transfer_learning
```
Note:

- While using transfer learning, make sure that you load weights into layers of the new model that have the same dimension with those of the pretrained model.

## 3. Test

- Testing the performance of the model stored in `model/210706-105857` on the test set `dataset/test`:
```
conda activate venv
python prediction_model.py --test --model_dir=model/210706-105857 --data_dir=dataset/test/data --data_list=dataset/test/fname.csv --batch_size=100 --save_result --plot_figure
```
## 4. Prediction

- Predicting arrival times for the prediction set `dataset/pred` by using the model in `model/210706-105857`:
```
conda activate venv
python prediction_model.py --model_dir=model/210706-105857 --data_dir=dataset/pred/data --data_list=dataset/pred/fname.csv --batch_size=100 --save_result --plot_figure
```
## 5. Correcting picks

In order to correct picks after predicting, go to the `correct_label` directory.
