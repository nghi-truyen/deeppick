import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import os
import time
import logging
from build_model import Model
from data_reader import Config, DataReader
from util import *
import pandas as pd
import random
import json

def read_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test",
                        action="store_true",
                        help="Do a test with labeled data")                     
    
    parser.add_argument("--batch_size",
                        default=100,
                        type=int,
                        help="batch size")
    
    parser.add_argument("--model_dir",
                        default=None,
                        help="Checkpoint directory (default: None)")
    
    parser.add_argument("--tp_prob",
                        default=0.3,
                        type=float,
                        help="Probability threshold for P pick")
    
    parser.add_argument("--ts_prob",
                        default=0.3,
                        type=float,
                        help="Probability threshold for S pick")
    
    parser.add_argument("--data_dir",
                        default="./dataset/test/data",
                        help="Input file directory")
    
    parser.add_argument("--data_list",
                        default="./dataset/test/fname.csv",
                        help="Input csv file")
    
    parser.add_argument("--output_dir",
                        default=None,
                        help="Output directory")
    
    parser.add_argument("--plot_figure",
                        action="store_true",
                        help="Plot figure for test or prediction")
                      
    parser.add_argument("--plot_rate",
                        default=1,
                        type=float,
                        help="Rate for plotting figures in test or prediction mode")
    
    parser.add_argument("--save_result",
                        action="store_true",
                        help="Save result for test or prediction")
    
    args = parser.parse_args()
    return args

def set_config(args):
    config = Config()
    
    config.X_shape = config.X_shape
    config.n_channel = config.X_shape[-1]
    config.Y_shape = config.Y_shape
    config.n_class = config.Y_shape[-1]
    config.dt = config.dt
    
    config.checkpoint_directory = args.model_dir
    config.batch_size = args.batch_size
    config.test = args.test
    if config.test:
        config.tol = config.tol
    config.tp_prob = args.tp_prob
    config.ts_prob = args.ts_prob
    
    return config

def prediction(args):
    config = set_config(args)
    if args.test:
        mode = 'test'
    else:
        mode = 'pred'
        args.save_result = True
    
    if args.save_result:  
        if args.output_dir == None:
            current_time = time.strftime("%y%m%d-%H%M%S")
            out_path = os.path.join('output', mode, current_time)
        else:
            out_path = args.output_dir
        logging.info("{}: {}".format(mode, out_path))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(os.path.join(out_path, 'config.log'), 'w') as fp:
            fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))
        result_dir = os.path.join(out_path, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if args.plot_figure:
            figure_dir = os.path.join(out_path, 'figures')
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
        else:
            figure_dir = None
    else:
        result_dir = None
        figure_dir = None
                
    df = pd.read_csv(args.data_list, header=0)
    logging.info("{} size: {}".format(mode,len(df)))
    
    # restore the model
    ## (This code allows to rebuild the model that fit the shape of prediction or test data by loading separately weights and rebuild the graph instead of loading graph and weights at the same time by using keras.models.load_model).
    try: # read parameters to rebuild the graph of model
        with open(os.path.join(os.path.split(args.model_dir)[0],'config.log')) as f_config:
            lines = f_config.readlines()
    except:
        try:
            with open(os.path.join(args.model_dir,'config.log')) as f_config:
                lines = f_config.readlines()
        except:
            logging.info("config file not found!")
            exit()
    keys = ['depths','filters_root','kernel_size','pool_size','dilation_rate','drop_rate'] # list of parameters to rebuild the model that fit the shape of the prediction (or test) data
    for key in keys:
        setattr(config, key, json.loads([string for string in lines if key in string][0][len(key)+2:]))
    try: # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
    except:
        pass
    logging.info("restoring model ...")    
    model = Model(config).get_model()  # build model with new graph  
    try: # load weights
        model.load_weights(args.model_dir)
    except:
        try:
            model.load_weights(os.path.join(args.model_dir,'model.h5'))
        except:
            logging.info("please set a right path for model_dir!")
            exit()
        
    # evaluate and predict
    model.compile(loss=[string for string in lines if 'loss_type' in string][0][len(key)+2:].rstrip())
    data = DataReader(mode=mode,data_dir=args.data_dir,df_list=df,batch_size=args.batch_size)
    if args.test: # evaluating only in test mode
        logging.info("evaluating loss:")
        evaluate = model.evaluate(data)
    logging.info("predicting...")
    preds = model.predict(data)
    
    # convert predicted results (detect peaks)
    picks = []
    itp = []
    its = []
    fname = df['fname'].tolist()
    for i in range(preds.shape[0]):
        pick = detect_peaks_all_classes(i, preds, fname, result_dir, args)
        picks += [pick]
        if args.test:
            fn_path = os.path.join(args.data_dir, fname[i])
            meta = np.load(fn_path)
            if type(meta['itp'].tolist()) == int:
                itp_tmp = [meta['itp'].tolist()]
            else:
                itp_tmp = meta['itp'].tolist()
            if type(meta['its'].tolist()) == int:
                its_tmp = [meta['its'].tolist()]
            else:
                its_tmp = meta['its'].tolist()
            itp += [itp_tmp]
            its += [its_tmp]
      
    if args.test: # calculate scoring metrics for test
        metrics_p, metrics_s = calculate_metrics(picks, itp, its)

    # save results
    if args.save_result:
        itp_pred = [item[0][0] for item in picks]
        its_pred = [item[1][0] for item in picks]
        itp_pred_prob = [item[0][1] for item in picks]
        its_pred_prob = [item[1][1] for item in picks]
        
        if args.test:
            # write score
            flog = open(os.path.join(out_path, 'metrics.log'), 'w')
            flog.write("loss: {}\n".format(evaluate))
            flog.write("P-phase: Precision={}, Recall={}, F1={}\n".format(metrics_p[0], metrics_p[1], metrics_p[2]))
            flog.write("S-phase: Precision={}, Recall={}, F1={}\n".format(metrics_s[0], metrics_s[1], metrics_s[2]))
            flog.close()
            
        # write picks into out file
            df_pick = pd.DataFrame({'fname':fname,'itp_true':itp,'itp_pred':itp_pred,'itp_pred_prob':itp_pred_prob,'its_true':its,'its_pred':its_pred,'its_pred_prob':its_pred_prob})
        else:
            df_pick = pd.DataFrame({'fname':fname,'itp_pred':itp_pred,'itp_pred_prob':itp_pred_prob,'its_pred':its_pred,'its_pred_prob':its_pred_prob})
        df_pick.to_csv(os.path.join(out_path, 'picks.csv'),index=False)   
        
        # store figures
        if figure_dir is not None:
            logging.info("plotting and storing figures...")
            for i in range(len(data)):
                if args.test:
                    X_tmp, Y_tmp = data[i] # get data on i_th batch
                else:
                    X_tmp = data[i] # get data on i_th batch
                    Y_tmp = None  
                for j in range(len(X_tmp)):
                    if random.random() < args.plot_rate:
                        plot_results(j,preds[i*args.batch_size:],X_tmp,Y_tmp,itp_pred[i*args.batch_size+j], its_pred[i*args.batch_size+j],fname[i*args.batch_size:],figure_dir)
        logging.info("results saved in {}".format(out_path))
    return 0

def main(args):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    prediction(args)
    return

if __name__ == '__main__':
  args = read_args()
  main(args)
