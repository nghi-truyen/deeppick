import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

class Config():
    n_channel = 1
    n_class = 3
    X_shape = [12001, 1, n_channel]
    Y_shape = [12001, 1, n_class]
    mask_window = int(X_shape[0]/100) # number of points for generating the distribution porobabilies of label 
    dt = 0.000025 # time derivatives
    tol = dt*mask_window/3 # acceptable uncertainty while testing
   
class DataReader(keras.utils.Sequence):
    def __init__(self,
               mode,
               data_dir, # type of str (path to data)
               df_list,  # type of dataframe
               batch_size,
               config=Config()):
        self.config = config
        self.df_list = df_list
        self.num_data = len(self.df_list)
        self.data_dir = data_dir
        self.mode = mode
        self.batch_size = batch_size
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.mask_window = config.mask_window + config.mask_window%2
        self.distribution_prob = np.exp(-(np.arange(-self.mask_window//2,self.mask_window//2))**2/(2*(self.mask_window//4)**2))
        self.buffer = {}
    def normalize(self, data):
        data -= np.mean(data, axis=0, keepdims=True)
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
        return data

    def adjust_missingchannels(self, data):
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
            data *= data.shape[-1] / np.count_nonzero(tmp)
        return data
    
    def __len__(self):
        return self.num_data//self.batch_size + 1*(self.num_data%self.batch_size>0)

    def __getitem__(self,idx): # get data on batch 
        """Returns tuple (input, target) correspond to batch #idx."""
        
        start = idx * self.batch_size
        index = list(range(start, min(self.num_data, start + self.batch_size)))
            
        X = np.zeros([len(index),]+self.X_shape)
        Y = np.zeros([len(index),]+self.Y_shape)
        
        if self.mode == 'train':
            np.random.shuffle(index)
        for j,i in enumerate(index):
            fname = os.path.join(self.data_dir, self.df_list.iloc[i]['fname'])
            try:
                if fname not in self.buffer:
                    meta = np.load(fname)
                    if self.mode == 'pred':
                        try:
                            self.buffer[fname] = {'data': meta['data']}
                        except:
                            logging.error("The column name must be correct!")
                            exit()
                    else:
                        try:
                            self.buffer[fname] = {'data': meta['data'], 'itp': meta['itp'], 'its': meta['its']}
                        except:
                            logging.error("The column names must be correct and the data must be labeled!")
                            exit()
                meta = self.buffer[fname]
            except:
                logging.error("Failed reading {}".format(fname))
                continue

            sample = np.zeros(self.X_shape)
            sample[:, :, :] = meta['data'][:self.X_shape[0], np.newaxis, :]
            sample = self.normalize(sample)
            sample = self.adjust_missingchannels(sample)
            X[j,:,:,:] = np.copy(sample)
            
            if not self.mode == 'pred':
                if type(meta['itp'].tolist()) == int:
                    itp_list = [meta['itp'].tolist()]
                else:
                    itp_list = meta['itp'].tolist()
                if type(meta['its'].tolist()) == int:
                    its_list = [meta['its'].tolist()]
                else:
                    its_list = meta['its'].tolist()
                
                target = np.zeros(self.Y_shape)
                for itp, its in zip(itp_list, its_list):
                    if (itp >= target.shape[0]) or (itp < 0):
                        pass
                    elif (itp-self.mask_window//2 >= 0) and (itp-self.mask_window//2 < target.shape[0]):
                        target[itp-self.mask_window//2:itp+self.mask_window//2, 0, 1] = self.distribution_prob[:target.shape[0]-(itp-self.mask_window//2)]
                    elif (itp-self.mask_window//2 < target.shape[0]):
                        target[0:itp+self.mask_window//2, 0, 1] = self.distribution_prob[:target.shape[0]-(itp-self.mask_window//2)]
                    if (its >= target.shape[0]) or (its < 0):
                        pass
                    elif (its-self.mask_window//2 >= 0) and (its-self.mask_window//2 < target.shape[0]):
                        target[its-self.mask_window//2:its+self.mask_window//2, 0, 2] = self.distribution_prob[:target.shape[0]-(its-self.mask_window//2)]
                    elif (its-self.mask_window//2 < target.shape[0]):
                        target[0:its+self.mask_window//2, 0, 2] = self.distribution_prob[:target.shape[0]-(its-self.mask_window//2)]
                target[:, :, 0] = 1 - target[:, :, 1] - target[:, :, 2]
                for i in range(len(target[:, :, 0])):
                    if target[:, :, 0][i] < 0:
                        target[:, :, 0][i] = 0
                Y[j,:,:,:] = np.copy(target)
        if self.mode == 'pred':
            return X
        else:
            return X,Y
