import json
import csv 
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import logging
from sklearn.linear_model import HuberRegressor

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
    
    parser.add_argument("--points_min_to_interpo",
                        default=6,
                        type=int,
                        help="Minumum of points considered to interpo")
                      
    args = parser.parse_args()
    return args

def log_poly_2(x,coef):
    return coef[0]*np.log(x)**2+coef[1]*np.log(x)+coef[2]

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
    
def model_p(X,y,eps,degree=2,flip=False): #logistic regression
    HR = HuberRegressor(epsilon=eps)
    reg = HR.fit(np.log(X).reshape(-1,1),y)
    mask = HR.outliers_  # detecting anomalies
    
    w = np.ones(len(y))
    w[mask] = 0
    coef = np.polyfit(np.log(X),y,degree,w=w)
    if flip:
        return coef, np.flip(mask)
    else:
        return coef, mask
    
def model_s(X,y,eps): # linear regression 
    HR = HuberRegressor(epsilon=eps)
    reg = HR.fit(X.reshape(-1,1),y)
    mask = HR.outliers_ # detecting anomalies
    return reg, mask

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
    
    if plot:
        if not os.path.exists('./figures'):
            os.makedirs('./figures')
            
    list_p_int = []
    list_s_int = []
    list_source = range(1,num_source+1)  #list_source = [1,2,50,51,52,92,93]
    logging.info('Correcting picks ...')
    for source in list_source:
        i=source-1
        labeltime = pd.read_csv(label_file)
        tp = np.copy(labeltime['itp_pred'][num_receiver*i:num_receiver*i+num_receiver])
        ts = np.copy(labeltime['its_pred'][num_receiver*i:num_receiver*i+num_receiver])

        r_p_left,list_p_left = pick_time(tp[:i],1,p_pick_min,p_pick_max)
        r_s_left,list_s_left = pick_time(ts[:i],1,s_pick_min,s_pick_max)
        r_p_right,list_p_right = pick_time(tp[i:],source,p_pick_min,p_pick_max)
        r_s_right,list_s_right = pick_time(ts[i:],source,s_pick_min,s_pick_max)

        r_p = np.append(r_p_left,r_p_right)
        r_s = np.append(r_s_left,r_s_right)
        list_p = np.append(list_p_left,list_p_right)
        list_s = np.append(list_s_left,list_s_right)
        
        y_p = np.zeros(num_receiver)
        y_s = np.zeros(num_receiver)
    
        if source<points_min_to_interpo:
        
            coef, mask_p = model_p(r_p_right, list_p_right, epsP)
            mask_p = np.array([False]*len(r_p_left)+list(mask_p))
            
            reg_s, mask_s = model_s(r_s_right, list_s_right,epsS)
            mask_s = np.array([False]*len(r_s_left)+list(mask_s))
            
            x = np.linspace(source,num_receiver,num_receiver-source+1)
            
            y_p[i:] = log_poly_2(x,coef)#reg_p.predict(np.log(x).reshape(-1,1))#
            y_s[i:] = reg_s.predict(x.reshape(-1,1))
            y_p[:i] = np.flip(np.copy(y_p[i+1:2*i+1]))
            y_s[:i] = np.flip(np.copy(y_s[i+1:2*i+1]))
            
            untreated_xp = r_p_left
            untreated_xs = r_s_left
            untreated_p = list_p_left
            untreated_s = list_s_left
            
        elif source<=num_receiver-points_min_to_interpo:
           
            coef1, mask_p1 = model_p(np.append(r_p_left,r_p_right[:1]), np.flip(np.append(list_p_left,list_p_right[:1])), epsP, flip=True)
            coef2, mask_p2 = model_p(r_p_right,list_p_right, epsP)
            
            reg_s1, mask_s1 = model_s(np.append(r_s_left,r_s_right[:1]), np.append(list_s_left,list_s_right[:1]),epsS)
            reg_s2, mask_s2 = model_s(r_s_right,list_s_right,epsS)
            
            mask_p = np.append(mask_p1[:-1],mask_p2)
            mask_s = np.append(mask_s1[:-1],mask_s2)
            
            x_right = np.linspace(source,num_receiver,num_receiver-source+1)
            x_left = np.linspace(1,source-1,source-1)

            y_p[i:] = log_poly_2(x_right,coef2) #reg_p1.predict(np.log(x_right).reshape(-1,1))#
            y_p[:i] = np.flip(log_poly_2(x_left,coef1)) #np.flip(reg_p2.predict(np.log(x_left).reshape(-1,1)))#
            y_s[i:] = reg_s2.predict(x_right.reshape(-1,1))
            y_s[:i] = reg_s1.predict(x_left.reshape(-1,1))
            
        else:
            coef, mask_p = model_p(r_p_left, np.flip(list_p_left), epsP, flip=True)
            mask_p = np.array(list(mask_p)+[False]*len(r_p_right))
            
            reg_s, mask_s = model_s(r_s_left, list_s_left,epsS)
            mask_s = np.array(list(mask_s)+[False]*len(r_s_right))
            
            x = np.linspace(1,source,source)

            y_p[:i+1] = np.flip(log_poly_2(x,coef)) #np.flip(reg_p.predict(np.log(x).reshape(-1,1)))#
            y_s[:i+1] = reg_s.predict(x.reshape(-1,1))
            y_p[i+1:] = np.flip(np.copy(y_p[2*i-num_receiver:i-1]))
            y_s[i+1:] = np.flip(np.copy(y_s[2*i-num_receiver:i-1]))  
   
            
            untreated_xp = r_p_right
            untreated_xs = r_s_right
            untreated_p = list_p_right
            untreated_s = list_s_right
                
        if plot:
            x=np.linspace(1,num_receiver,num_receiver)
            plt.figure(i)
            plt.subplot(111)
            plt.xlabel('Receiver')
            plt.ylabel('Arrival time')
            plt.scatter(r_p,list_p,s=15)
            if (source>1 and source<points_min_to_interpo) or (source<96 and source>num_receiver-points_min_to_interpo):
                plt.scatter(untreated_xp,untreated_p,c='orange',label='Untreated points',s=15)
            plt.scatter(np.array(r_p)[mask_p],np.array(list_p)[mask_p],c='red',label='Outliers, epsP={}'.format(epsP),s=15)       
            plt.plot(x,y_p,label='P corrected picks',c='green')
            plt.legend()
            plt.savefig('./figures/Source {}_P'.format(source))
            plt.close(i)
            plt.figure(i)
            plt.subplot(111)
            plt.xlabel('Receiver')
            plt.ylabel('Arrival time')
            plt.scatter(r_s,list_s,s=15)
            if (source>1 and source<points_min_to_interpo) or (source<96 and source>num_receiver-points_min_to_interpo):
                plt.scatter(untreated_xs,untreated_s,c='orange',label='Untreated points',s=15)
            plt.scatter(np.array(r_s)[mask_s],np.array(list_s)[mask_s],c='r',label='Outliers, epsS={}'.format(epsS),s=15)     
            plt.plot(x,y_s,label='S corrected picks',c='green')
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
