from __future__ import division
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from data_reader import Config
from detect_peaks import detect_peaks
import logging

def detect_peaks_all_classes(i, pred, fname=None, result_dir=None, args=None):
  distance = Config().Y_shape[0] # allow to return only the maximum peak probability
  if args is None:
    itp = detect_peaks(pred[i,:,0,1], mph=0.5, mpd=distance, show=False)
    its = detect_peaks(pred[i,:,0,2], mph=0.5, mpd=distance/Config().dt, show=False)
  else:
    itp = detect_peaks(pred[i,:,0,1], mph=args.tp_prob, mpd=distance, show=False)
    its = detect_peaks(pred[i,:,0,2], mph=args.ts_prob, mpd=distance, show=False)
  prob_p = pred[i,itp,0,1]
  prob_s = pred[i,its,0,2]
  if (fname is not None) and (result_dir is not None):
    try:
      np.savez(os.path.join(result_dir, fname[i]), pred=pred[i], itp=itp, its=its, prob_p=prob_p, prob_s=prob_s)
    except FileNotFoundError:
      os.makedirs(os.path.dirname(os.path.join(result_dir, fname[i])), exist_ok=True)
      np.savez(os.path.join(result_dir, fname[i]), pred=pred[i], itp=itp, its=its, prob_p=prob_p, prob_s=prob_s)
  return [(itp, prob_p), (its, prob_s)]

def plot_results(i, pred, X, Y=None, itp_pred=None, its_pred=None, fname=None, figure_dir=None):
  dt = Config().dt
  t = np.arange(0, pred.shape[1]) * dt
  box = dict(boxstyle='round', facecolor='white', alpha=1)
  text_loc = [0.05, 0.77]
  plt.figure(i)
  plt.subplot(211)
  plt.plot(t, X[i, :, 0, 0], 'k', label='channel_Z', linewidth=0.5)
  plt.autoscale(enable=True, axis='x', tight=True)
  tmp_min = np.min(X[i, :, 0, 0])
  tmp_max = np.max(X[i, :, 0, 0])
  if (itp_pred is not None) and (its_pred is not None):
    for j in range(len(itp_pred)):
      plt.plot([itp_pred[j]*dt, itp_pred[j]*dt], [tmp_min, tmp_max], '--g', linewidth=0.5)
    for j in range(len(its_pred)):
      plt.plot([its_pred[j]*dt, its_pred[j]*dt], [tmp_min, tmp_max], '-.m', linewidth=0.5)
  plt.ylabel('Amplitude')
  plt.legend(loc='upper right', fontsize='small')
  plt.gca().set_xticklabels([])
  plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

  plt.subplot(212)
  if Y is not None:
    plt.plot(t, Y[i, :, 0, 1], 'b', label='P', linewidth=0.5)
    plt.plot(t, Y[i, :, 0, 2], 'r', label='S', linewidth=0.5)
  plt.plot(t, pred[i, :, 0, 1], '--g', label='$\hat{P}$', linewidth=0.5)
  plt.plot(t, pred[i, :, 0, 2], '-.m', label='$\hat{S}$', linewidth=0.5)
  plt.autoscale(enable=True, axis='x', tight=True)
  if (itp_pred is not None) and (its_pred is not None):
    for j in range(len(itp_pred)):
      plt.plot([itp_pred[j]*dt, itp_pred[j]*dt], [-0.1, 1.1], '--g', linewidth=0.5)
    for j in range(len(its_pred)):
      plt.plot([its_pred[j]*dt, its_pred[j]*dt], [-0.1, 1.1], '-.m', linewidth=0.5)
  plt.ylim([-0.05, 1.05])
  plt.text(text_loc[0], text_loc[1], '(iv)', horizontalalignment='center',
           transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
  plt.legend(loc='upper right', fontsize='small')
  plt.xlabel('Time (s)')
  plt.ylabel('Probability')

  plt.tight_layout()
  plt.gcf().align_labels()

  try:
    plt.savefig(os.path.join(figure_dir, 
                fname[i].rstrip('.npz')+'.png'), 
                bbox_inches='tight')
  except FileNotFoundError:
    os.makedirs(os.path.dirname(os.path.join(figure_dir, fname[i])), exist_ok=True)
    plt.savefig(os.path.join(figure_dir, 
                fname[i].rstrip('.npz')+'.png'), 
                bbox_inches='tight')
  plt.close(i)
  return 0

def metrics(TP, nP, nT):
  '''
  TP: true positive
  nP: number of positive picks
  nT: number of true picks
  '''
  precision = TP / nP
  recall = TP / nT
  F1 = 2* precision * recall / (precision + recall)
  return [precision, recall, F1]

def correct_picks(picks, true_p, true_s):
  dt = Config().dt
  tol = Config().tol
  if len(true_p) != len(true_s):
    print("The length of true P and S pickers are not the same")
  num = len(true_p)
  TP_p = 0; TP_s = 0; nP_p = 0; nP_s = 0; nT_p = 0; nT_s = 0
  for i in range(num):
    nT_p += len(true_p[i])
    nT_s += len(true_s[i])
    nP_p += len(picks[i][0][0])
    nP_s += len(picks[i][1][0])

    if len(true_p[i]) > 1 or len(true_s[i]) > 1:
      print(i, picks[i], true_p[i], true_s[i])
    tmp_p = np.array(picks[i][0][0]) - np.array(true_p[i])[:,np.newaxis]
    tmp_s = np.array(picks[i][1][0]) - np.array(true_s[i])[:,np.newaxis]
    TP_p += np.sum(np.abs(tmp_p) < tol/dt)
    TP_s += np.sum(np.abs(tmp_s) < tol/dt)

  return [TP_p, TP_s, nP_p, nP_s, nT_p, nT_s]

def calculate_metrics(picks, itp, its):
  TP_p, TP_s, nP_p, nP_s,  nT_p, nT_s = correct_picks(picks, itp, its)
  precision_p, recall_p, f1_p = metrics(TP_p, nP_p, nT_p)
  precision_s, recall_s, f1_s = metrics(TP_s, nP_s, nT_s)
  
  logging.info("Total records: {}".format(len(picks)))
  logging.info("P-phase:")
  logging.info("True={}, Predict={}, TruePositive={}".format(nT_p, nP_p, TP_p))
  logging.info("Precision={:.3f}, Recall={:.3f}, F1={:.3f}".format(precision_p, recall_p, f1_p))
  logging.info("S-phase:")
  logging.info("True={}, Predict={}, TruePositive={}".format(nT_s, nP_s, TP_s))
  logging.info("Precision={:.3f}, Recall={:.3f}, F1={:.3f}".format(precision_s, recall_s, f1_s))
  return [precision_p, recall_p, f1_p], [precision_s, recall_s, f1_s]
