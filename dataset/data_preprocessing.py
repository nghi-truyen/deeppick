import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import fnmatch
import random
import logging

class Config():
    file_type = '.semv' # Format of raw data files
    data_size = 10001 # chose a output size for data that will be used for training, test or prediction.
    noise_level = [0.01,0.04] # Minimum and maximum of noise levels while doing data augmentation.
    expand_dim = 3 # Increase the size of train set (or test set) with a factor of expand_dim while training (or testing) on each batch.

def adding_noise(freq,SNR):
    RMS = np.sqrt(np.mean(freq**2))
    deviance = RMS/np.sqrt(10**(SNR/10))
    return freq+np.random.normal(0,deviance,len(freq))
    
def convert_to_npz(args,out_dir):
    config = Config()
    if args.mode == 'train' or args.mode == 'test':
        label = True
    elif args.mode == 'pred':
        label = False
    else:
        logging.info("mode must be train, test or pred")
        exit()
    list_data = np.sort(fnmatch.filter(os.listdir(args.data_dir), '*{}'.format(config.file_type))).tolist()
    try:
        df = pd.read_csv(args.label_list)
        itp = df['itp'].to_numpy()
        its = df['its'].to_numpy()
        itray = df['itray'].to_numpy()
    except:
        if not label:
            pass
        else:
            logging.info("Unlabeled data! Train or test mode need a labeled data.")
            exit()
    
    fname_list = []
    itp_list = []
    its_list = []
    itray_list = []
    for i,filename in enumerate(list_data):
        # Load data from text file
        seismo = np.loadtxt(os.path.join(args.data_dir,filename))
        freq = seismo[:,1]
        n = len(freq)
        # Write and save to .npz file
        data = freq.reshape(n,1)
        if not label:
            filename_npz = os.path.join(out_dir,'data',filename.rstrip(config.file_type))
            np.savez_compressed(filename_npz,data=data[:config.data_size])
            fname_list += [filename.rstrip(config.file_type)+'.npz']
        else:
            # create a database for training or testing
            expand_dim = config.expand_dim*args.data_augmentation + 1*(not args.data_augmentation)
            for j in range(expand_dim):
                filename_npz = os.path.join(out_dir,'data',filename.rstrip(config.file_type)+'_{}'.format(j))
                if random.random() > 0.05 or its[i]+int(config.data_size/10) > n-config.data_size:
                    if j == 0:
                        shift = 0
                        coef = 1
                    elif j == 1:
                        shift = 0
                        coef = -1 
                    else:
                        if max(0,its[i]-config.data_size+int(config.data_size/10)) < itp[i]-int(config.data_size/100):
                            shift = random.randint(max(0,its[i]-config.data_size+int(config.data_size/10)), itp[i]-int(config.data_size/100))
                            coef = (-1)**(random.random()>0.5)
                        else:
                            logging.info("Can not generate file {} because of small output data size (please increase the value of data_size if you want to create a completed database)".format(filename_npz))                       
                            break
                    [SNR_max,SNR_min] = 1/np.array(config.noise_level)
                    SNR = random.randint(SNR_min, SNR_max)
                    data_ = data[shift:shift+config.data_size]*coef
                    data_[:,0] = adding_noise(data_[:,0],SNR)
                    itp_ = itp[i]-shift
                    its_ = its[i]-shift
                    itray_ = itray[i]-shift
                    if itp_ > config.data_size:
                        itp_ = []
                    if its_ > config.data_size:
                        its_ = []
                    if itray_ > config.data_size:
                        itray_ = []
                else:
                    shift = random.randint(its[i]+int(config.data_size/10), n-config.data_size)
                    data_ = data[shift:shift+config.data_size]
                    itp_ = []
                    its_ = []
                    itray_ = []
                np.savez_compressed(filename_npz,data=data_,itp=itp_,its=its_,itray=itray_)
                fname_list += [filename.rstrip(config.file_type)+'_{}{}'.format(j,'.npz')]
                itp_list += [itp_]
                its_list += [its_]
                itray_list += [itray_]
    
    # Create a corresponding csv file
    if not label:
        dataframe = pd.DataFrame({'fname':fname_list})
    else:
        dataframe = pd.DataFrame({'fname':fname_list,'itp':itp_list,'its':its_list,'itray':itray_list})
    dataframe.to_csv(os.path.join(out_dir,'fname.csv'),index=False)
    
    return n, len(list_data), config.data_size, len(dataframe)
