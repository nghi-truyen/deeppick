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
where `mode` is either `train` (for generating train test), or `test` (for generating test set). The action `--data_augmentation` used in order to increase the size of train or test set. We can further add the action `--plot_figure` and set a value for the parameter `--plot_rate` in order to verify the data preprocessing results.

- For generating unlabeled data (for prediction):
```
conda activate venv
python run.py --mode=pred --data_dir=raw/data
```
## 2. Training
Now, go back to the main directory.
### Training from scratch:

- Train by splitting data into train and valid set: 
```
conda activate venv
python train_model.py --valid --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --optimizer=adam
```
- Train on the whole dataset:
```
conda activate venv
python train_model.py --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --optimizer=adam
```
### Transfer learning:

- Train with the validation set and using transfer learning (initializing weights from a pretrained model). By default, the process will load weights from all layers of pretrained model and fine-tune weights in all of these layers: 
```
conda activate venv
python train_model.py --valid --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --model_dir=model/pretrained_from_NCEDC
```
- For more options of this process, you can see the function `ind_layers()` in `train_model.py` as well as the summary file of pretrained model in order to chose the index of layers for loading and freezing weights when using transfer learning. In this scenario, you need to modify the `TO IMPLEMENT` part in this function and set the action `--tune_transfer_learning` as the command below:
```
conda activate venv
python train_model.py --valid --data_dir=dataset/train/data --data_list=dataset/train/fname.csv --batch_size=100 --epochs=50 --model_dir=model/pretrained_from_NCEDC --tune_transfer_learning
```
## 3. Test
```
conda activate venv
python prediction_model.py --test --model_dir=model/210520-204441 --data_dir=dataset/test/data --data_list=dataset/test/fname.csv --batch_size=100 --save_result --plot_figure
```
## 4. Prediction
```
conda activate venv
python prediction_model.py --model_dir=model/210520-204441 --data_dir=dataset/pred/data --data_list=dataset/pred/fname.csv --batch_size=100 --save_result --plot_figure
```
Notes:

- In the data preprocessing, the arrival times in `dataset/raw/label_time.csv` must be sorted in ascending alphabetical order of the file name in the `dataset/raw/data` directory.

- For predicting and testing, the shape of data in train (or test) directory and that of the model shown in `config.log` must have the same dimension.
