import json
import csv 
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import logging
from sklearn.linear_model import HuberRegressor

from sklearn.svm import SVR

def read_args():
    parser = argparse.ArgumentParser()
                  
    parser.add_argument("--picks_file",
                        default="./picks.csv",
                        help="Picks file")
    
    parser.add_argument("--num_source",
                        default=None,
                        type=int,
                        help="Number of sources")
    
    parser.add_argument("--num_receiver",
                        default=None,
                        type=int,
                        help="Number of receiver")
                          
    parser.add_argument("--output_name",
                        default="label_time.csv",
                        help="Output label time name")
    
    parser.add_argument("--plot_figure",
                        action="store_true",
                        help="Plot figures")
    
    parser.add_argument("--p_pick_min",
                        default=None,
                        type=int,
                        help="Lower bound of P that helps to dectect anomalies")
    
    parser.add_argument("--p_pick_max",
                        default=None,
                        type=int,
                        help="Upper bound of P that helps to dectect anomalies")
    
    parser.add_argument("--s_pick_min",
                        default=None,
                        type=int,
                        help="Lower bound of S that helps to dectect anomalies")
    
    parser.add_argument("--s_pick_max",
                        default=None,
                        type=int,
                        help="Upper bound of S that helps to dectect anomalies")
                        
    parser.add_argument("--epsP",
                        default=1.35,
                        type=float,
                        help="The parameter controls the number of samples that should be classified as outliers for P while using Huber Regression. The smaller the epsilon, the more robust it is to outliers.")
    
    parser.add_argument("--epsS",
                        default=1.35,
                        type=float,
                        help="The parameter controls the number of samples that should be classified as outliers for S while using Huber Regression. The smaller the epsilon, the more robust it is to outliers.")
    
    parser.add_argument("--CP",
                        default=1,
                        type=float,
                        help="Regularization parameter for P while using SVR. The strength of the regularization is inversely proportional to CP.")
    
    parser.add_argument("--CS",
                        default=1,
                        type=float,
                        help="Regularization parameter for S while using SVR. The strength of the regularization is inversely proportional to CS.")
    
    parser.add_argument("--points_min_to_interpo",
                        default=8,
                        type=int,
                        help="Minumum of points considered to interpo")
                      
    args = parser.parse_args()
    return args

def pick_time(t,r_start,pick_min,pick_max):
    it = [json.loads(t[i]) for i in range(len(t))]
    r = []
    list_t = []
    for i in range(len(it)):
        try:
            if it[i][0]<pick_max and it[i][0]>pick_min: 
                list_t += [it[i][0]]
                r += [i+r_start]
        except:
            pass
    return np.array(r),np.array(list_t)
    
def model(wave,X,y,source,eps,C,num_recei,min_point_to_interpo): 
    
    if source < num_recei-min_point_to_interpo:
        found = False
        i = source
        while not found and i <= num_recei:
            try:
                ind = np.where(X==i)[0][0]
                found = True
            except:
                i+=1
        HR = HuberRegressor(epsilon=eps)
        if wave=='P':
            HR.fit(np.log(X[ind:]).reshape(-1,1),y[ind:])
        elif wave=='S':
            HR.fit(X[ind:].reshape(-1,1),y[ind:])
        mask = HR.outliers_
        r = True
    else:
        r = False
    if source > min_point_to_interpo:
        found_ = False
        i_ = source
        while not found_ and i_ > 0:
            try:
                ind_ = np.where(X==i_)[0][0]
                found_ = True
            except:
                i_-=1
        HR_ = HuberRegressor(epsilon=eps)
        if wave=='P':
            HR_.fit(np.log(X[:ind_+1]).reshape(-1,1),np.flip(y[:ind_+1]))
            mask_ = np.flip(HR_.outliers_)
        elif wave=='S':
            HR_.fit(X[:ind_+1].reshape(-1,1),y[:ind_+1])
            mask_ = HR_.outliers_
        l = True
        if i_ == source:
            mask_ = mask_[:-1]
    else:
        l = False
    if r==False:
        mask = np.array([2]*(len(X)-len(mask_)))
    if l==False:
        mask_ = np.array([2]*(len(X)-len(mask)))
        
    mask = np.append(mask_,mask)
    reg = SVR(kernel='rbf', C=C).fit(X[mask==False].reshape(-1,1),y[mask==False])
      
    return reg,mask,l,r

def correct_label(args):
    
    label_file = args.picks_file
    num_source = args.num_source
    num_receiver = args.num_receiver
    outname = args.output_name
    plot = args.plot_figure
    points_min_to_interpo = args.points_min_to_interpo
    p_pick_min = args.p_pick_min
    s_pick_min = args.s_pick_min
    p_pick_max = args.p_pick_max
    s_pick_max = args.s_pick_max
    epsP = args.epsP
    epsS = args.epsS
    CP = args.CP
    CS = args.CS
    
    if plot:
        if not os.path.exists('./figures'):
            os.makedirs('./figures')
        logging.info('Correcting picks and plotting...')
    else:
        logging.info('Correcting picks ...')
            
    list_p_int = []
    list_s_int = []
    list_source = range(1,num_source+1)  #list_source = [1,2,50,51,52,92,93]
    for j,source in enumerate(list_source):
        i=source-1
        labeltime = pd.read_csv(label_file)
        tp = np.copy(labeltime['itp_pred'][num_receiver*j:num_receiver*j+num_receiver])
        ts = np.copy(labeltime['its_pred'][num_receiver*j:num_receiver*j+num_receiver])

        r_p,list_p = pick_time(tp,1,p_pick_min,p_pick_max)
        r_s,list_s = pick_time(ts,1,s_pick_min,s_pick_max)
        
        y_p = np.zeros(num_receiver)
        y_s = np.zeros(num_receiver)
        
        reg_p, mask_p, lp, rp = model('P',r_p, list_p, source,epsP,CP,num_receiver,points_min_to_interpo)
        reg_s, mask_s, ls, rs = model('S',r_s, list_s,source,epsS,CS,num_receiver,points_min_to_interpo)
        
        x = np.linspace(1,num_receiver,num_receiver)
  
        if lp and rp:
            y_p = reg_p.predict(x.reshape(-1,1))
        elif not lp:
            y_p[i:] = reg_p.predict(x[i:].reshape(-1,1))
            y_p[:i] = np.flip(np.copy(y_p[i+1:2*i+1]))
        elif not rp:
            y_p[:i+1] = reg_p.predict(x[:i+1].reshape(-1,1)) 
            y_p[i+1:] = np.flip(np.copy(y_p[2*i-num_receiver:i-1]))
        if ls and rs:
            y_s = reg_s.predict(x.reshape(-1,1))
        elif not ls:
            y_s[i:] = reg_s.predict(x[i:].reshape(-1,1))
            y_s[:i] = np.flip(np.copy(y_s[i+1:2*i+1]))
        elif not rs:
            y_s[:i+1] = reg_s.predict(x[:i+1].reshape(-1,1)) 
            y_s[i+1:] = np.flip(np.copy(y_s[2*i-num_receiver:i-1]))
                
        if plot:
            plt.figure(i)
            plt.subplot(111)
            plt.xlabel('Receiver')
            plt.ylabel('Arrival time')
            plt.scatter(r_p,list_p,s=15)
            if (source>1 and source<points_min_to_interpo) or (source<num_receiver and source>num_receiver-points_min_to_interpo):
                plt.scatter(np.array(r_p)[mask_p==2],np.array(list_p)[mask_p==2],c='orange',label='Untreated points',s=15)
            plt.scatter(np.array(r_p)[mask_p==True],np.array(list_p)[mask_p==True],c='red',label='Outliers, epsP={}'.format(epsP),s=15)       
            plt.plot(x,y_p,label='P corrected picks, CP={}'.format(CP),c='green')
            plt.legend()
            plt.savefig('./figures/Source {}_P'.format(source))
            plt.close(i)
            plt.figure(i)
            plt.subplot(111)
            plt.xlabel('Receiver')
            plt.ylabel('Arrival time')
            plt.scatter(r_s,list_s,s=15)
            if (source>1 and source<points_min_to_interpo) or (source<num_receiver and source>num_receiver-points_min_to_interpo):
                plt.scatter(np.array(r_s)[mask_s==2],np.array(list_s)[mask_s==2],c='orange',label='Untreated points',s=15)
            plt.scatter(np.array(r_s)[mask_s==True],np.array(list_s)[mask_s==True],c='r',label='Outliers, epsS={}'.format(epsS),s=15)     
            plt.plot(x,y_s,label='S corrected picks, CS={}'.format(CS),c='green')
            plt.legend()
            plt.savefig('./figures/Source {}_S'.format(source))
            plt.close(i)

        p_int = [round(j) for j in y_p]
        s_int = [round(j) for j in y_s]
        list_p_int += p_int
        list_s_int += s_int
        
    if plot:
        logging.info('Figures stored in {}'.format('figures')) 
    dataframe = pd.DataFrame({'itp':list_p_int,'its':list_s_int})
    dataframe.to_csv(outname,index=False)
    logging.info('Label times are saved in {}'.format(outname))

def main(args):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    correct_label(args)
    return

if __name__ == '__main__':
  args = read_args()
  main(args)
