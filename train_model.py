import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import logging
from tensorflow.keras import layers
import time
import random
import argparse
from build_model import Model
from data_reader import Config, DataReader
from contextlib import redirect_stdout


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--valid",
                        action="store_true",
                        help="Split into train and valid set")
                        
    parser.add_argument("--epochs",
                        default=60,
                        type=int,
                        help="number of epochs")
    
    parser.add_argument("--batch_size",
                        default=100,
                        type=int,
                        help="batch size")
    
    parser.add_argument("--learning_rate",
                        default=0.01,
                        type=float,
                        help="learning rate")
    
    parser.add_argument("--filters_root",
                        default=8,
                        type=int,
                        help="filters root")
    
    parser.add_argument("--depth",
                        default=5,
                        type=int,
                        help="depth")
    
    parser.add_argument("--kernel_size",
                        nargs="+",
                        type=int,
                        default=[7, 1],
                        help="kernel size")
    
    parser.add_argument("--pool_size",
                        nargs="+",
                        type=int,
                        default=[4, 1],
                        help="pool size")
    
    parser.add_argument("--drop_rate",
                        default=0,
                        type=float,
                        help="drop out rate")
    
    parser.add_argument("--dilation_rate",
                        nargs="+",
                        type=int,
                        default=[1, 1],
                        help="dilation rate")
    parser.add_argument("--optimizer",
                        default="Adam",
                        help="optimizer: Adam, Adagrad, Nadam, RMSprop")
                        
    parser.add_argument("--loss_type",
                        default="categorical_crossentropy",
                        help="loss type: categorical_crossentropy,...")
    
    parser.add_argument("--data_dir",
                        default="./dataset/train/data",
                        help="Input file directory")
    
    parser.add_argument("--data_list",
                        default="./dataset/train/fname.csv",
                        help="Input csv file")

    # parameters for transfer learning                    
    parser.add_argument("--model_dir",
                        default=None,
                        help="model directory used for transfer learning. If None: training from scratch (default: None)")

    parser.add_argument("--tune_transfer_learning",
                        action="store_true",
                        help="Change default parameters of transfer learning process (see the function ind_layers() for more details)")

    args = parser.parse_args()
    return args
    
def set_config(args):
    config = Config()
    
    config.X_shape = config.X_shape
    config.n_channel = config.X_shape[-1]
    config.Y_shape = config.Y_shape
    config.n_class = config.Y_shape[-1]
    
    config.depths = args.depth
    config.filters_root = args.filters_root
    config.kernel_size = args.kernel_size
    config.pool_size = args.pool_size
    config.dilation_rate = args.dilation_rate
    config.batch_size = args.batch_size
    config.optimizer = args.optimizer
    config.loss_type = args.loss_type
    config.learning_rate = args.learning_rate
    config.drop_rate = args.drop_rate
    if args.model_dir is None:
        config.transfer_learning = False
    else:    
        config.transfer_learning = True
        config.model_dir = args.model_dir
    return config
    
def ind_layers(args,n_layers): # return index of layers for loading and freezing weights when using transfer learning (see summary file of pretrained model for more details, that allows to determine these index by visualizing the architec of model)

    if not args.tune_transfer_learning:
    
        ## DEFAULT ##
        ind_load = list(range(0,n_layers)) # load weights from all layers of pretrained model
        ind_freeze = [] # fine-tuning all layers
        ##         ##
        
    else:
    
        # ind_load = list(range(start_load,end_load)) # loading weights only from some deep layers (the other weights are randomly initialized)
        # ind_freeze = list(range(start_freeze,end_freeze)) # freezing some deep layers
        
        ## TO IMPLEMENT ##
        ind_load = list(range(0,41))  # loading weights only from input and encoder layers
        ind_freeze = list(range(0,41)) # freezing input and encoder layers
        ##              ##
        
    return ind_load,ind_freeze,n_layers

def train_fn(args, train_ratio=0.9):
    current_time = time.strftime("%y%m%d-%H%M%S")
    model_path = os.path.join('model', current_time)
    logging.info("Training: {}".format(model_path))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    config = set_config(args)
    with open(os.path.join(model_path, 'config.log'), 'w') as fp:
        fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))
    df = pd.read_csv(args.data_list, header=0)
    df = df.iloc[np.random.permutation(len(df))]
    logging.info("Total training size: {}".format(len(df)))
    
    try: # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
    except:
        pass
        
    # Build model
    model = Model(config).get_model()
    
    # transfer learning
    if args.model_dir is not None: 
        logging.info("restoring model for transfer learning ...")
        try: # restore the model
            pretrain_model = keras.models.load_model(args.model_dir)
        except:
            try:
                pretrain_model = keras.models.load_model(os.path.join(args.model_dir,'model.h5'))
            except:
                logging.info("please set a right path for model_dir!")
                exit()
        pretrain_layers = pretrain_model.layers
        ind_load,ind_freeze,n_layers = ind_layers(args,len(pretrain_layers))
        for i in ind_load: # get weights from some (or all) layers of pretrained model
            model.layers[i].set_weights(pretrain_layers[i].get_weights())
        for i in ind_freeze: # freezing some layers of new model
            model.layers[i].trainable = False
        # add config for transfer learning         
        with open(os.path.join(model_path, 'config.log'), 'a') as fp_add:
            fp_add.write('\n')
            fp_add.write('total_layers: {}\n'.format(n_layers))
            fp_add.write('loaded_weights_layers: {}\n'.format(ind_load))
            fp_add.write('frozen_weights_layers: {}'.format(ind_freeze))
    
    # write summary
    with open(os.path.join(model_path, 'summary'), 'w') as f_sum:
        with redirect_stdout(f_sum):
            model.summary() 
         	
    # Set optimizer
    if config.optimizer == 'adam' or config.optimizer == 'Adam':
    	opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    elif config.optimizer == 'adagrad' or config.optimizer == 'Adagrad':
    	opt = keras.optimizers.Adagrad(learning_rate=config.learning_rate)
    elif config.optimizer == 'nadam' or config.optimizer == 'Nadam':
    	opt = keras.optimizers.Nadam(learning_rate=config.learning_rate)
    elif config.optimizer == 'rmsprop' or config.optimizer == 'RMSprop':
    	opt = keras.optimizers.RMSprop(learning_rate=config.learning_rate)
    else:
    	logging.info("This type of optimizer is not set yet!")
    	exit()
    	
    # Compile model	
    model.compile(loss=args.loss_type, optimizer=opt)
    callbacks = [keras.callbacks.ModelCheckpoint("{}/model.h5".format(model_path), save_best_only=args.valid)]
    epoch_list = range(1,args.epochs+1)
    
    # Split into train and valid set or not
    if not args.valid:
        logging.info("Training on the whole train set ... ")
        data_train = DataReader(mode='train',data_dir=args.data_dir,df_list=df,batch_size=args.batch_size)
        history = model.fit(data_train, epochs=args.epochs, callbacks=callbacks)
        train_loss = history.history['loss']
        df_loss = pd.DataFrame({'epoch':epoch_list,'train_loss':train_loss})
        df_loss.to_csv("{}/loss.csv".format(model_path),index=False)
        logging.warning("Stored lastest checkpoint in {}/model.h5 (best model can be saved only in validation mode)".format(model_path))
    else:
        logging.info("Split into train and valid set ... ")
        ind_drop_train = []
        ind_drop_valid = []
        for i in range(len(df)):
            if random.uniform(0,1) < train_ratio:
                ind_drop_valid += [i]
            else:
                ind_drop_train += [i]
        df_train = df.drop(ind_drop_train,axis=0)
        df_valid = df.drop(ind_drop_valid,axis=0)
        data_train = DataReader(mode='train',data_dir=args.data_dir,df_list=df_train,batch_size=config.batch_size)
        data_valid = DataReader(mode='train',data_dir=args.data_dir,df_list=df_valid,batch_size=config.batch_size)
        logging.info("Train size: {}".format(len(df_train)))
        logging.info("Valid size: {}".format(len(df_valid)))
        history = model.fit(data_train, epochs=args.epochs,validation_data=data_valid, callbacks=callbacks)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        df_loss = pd.DataFrame({'epoch':epoch_list,'train_loss':train_loss,'val_loss':val_loss})
        df_loss.to_csv("{}/loss.csv".format(model_path),index=False)
        logging.info("Stored best model in {}/model.h5".format(model_path))

    return 0

def main(args):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    train_fn(args)
    return

if __name__ == '__main__':
  args = read_args()
  main(args)
