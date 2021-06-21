import numpy as np
from data_preprocessing import convert_to_npz
import argparse
import logging
import random
import matplotlib.pyplot as plt
import os

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        default=None,
                        help="Mode of preprocessing (must be 'train', 'test' or 'pred')")
                        
    parser.add_argument("--data_augmentation",
                        action="store_true",
                        help="Increase the size of train (or test) set with a factor of expand_dim while training(or testing) on each batch (see in class Config() of data_preprocessing.py)")
                                            
    parser.add_argument("--data_dir",
                        default="./raw/data",
                        help="Input file directory of raw data")
    
    parser.add_argument("--label_list",
                        default=None,
                        help="Input label time file (for train or test mode)")
                        
    parser.add_argument("--output_dir",
                        default=None,
                        help="Output directory")
    
    parser.add_argument("--plot_figure",
                        action="store_true",
                        help="Plot figure for visualizing generated data files")
                      
    parser.add_argument("--plot_rate",
                        default=1,
                        type=float,
                        help="Rate for plotting figures")
                        
    args = parser.parse_args()
    return args
    
def database(args):
    logging.info('Preprocessing data ...')
    if args.output_dir == None:
        out_dir = args.mode
    else:
        out_dir = args.output_dir
    if not os.path.exists(os.path.join(out_dir,'data')):
        os.makedirs(os.path.join(out_dir,'data'))
    size_in, n_in, size_out, n_out = convert_to_npz(args,out_dir)
    logging.info('Size of raw dataset: {}. Size of each sample of raw data: {}'.format(n_in,size_in))
    logging.info('Mode: {}'.format(args.mode))
    logging.info('Data augmentation (only for train and test mode): {}'.format(args.data_augmentation))
    logging.info('Size of preprocessed dataset: {}. Size of each sample of preprocessed data: {}'.format(n_out,size_out))
    if args.plot_figure: # if want to verify and visualize results
        logging.info('Plotting and storing figures ...')
        x = np.linspace(0,size_out-1,size_out)
        figure_dir = os.path.join(out_dir,'figures')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        for i,filename in enumerate(os.listdir(os.path.join(out_dir,'data'))):
            if filename.endswith('.npz'):
                if random.random()<args.plot_rate:
                    data = np.load(os.path.join(out_dir,'data',filename))
                    plt.figure(i)
                    plt.subplot(111)
                    plt.xlabel('Time (sample)')
                    plt.ylabel('Amplitude')
                    plt.plot(x,data['data'][:,0],label='channel_Z',linewidth=0.5)
                    try:
                        plt.axvline(data['itp'],c='red',label='P',linewidth=0.5)
                        plt.axvline(data['its'],c='green',label='S',linewidth=0.5)
#                        plt.axvline(data['itray'],c='blue',label='Rayleigh',linewidth=0.5)
                    except:
                        pass
                    plt.legend(loc='upper right', fontsize='small')
                    # Save figures
                    try:
                        plt.savefig(os.path.join(figure_dir, filename.rstrip('.npz')+'.png'), bbox_inches='tight')
                    except FileNotFoundError:
                        os.makedirs(os.path.dirname(os.path.join(figure_dir, filename)), exist_ok=True)
                        plt.savefig(os.path.join(figure_dir, filename.rstrip('.npz')+'.png'), bbox_inches='tight')
                    plt.close(i)
    logging.info('{} set is stored in {}'.format(args.mode, out_dir))
 
def main(args):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    database(args)
    return

if __name__ == '__main__':
  args = read_args()
  main(args)
