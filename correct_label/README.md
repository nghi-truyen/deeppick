Correcting predicted picks in `picks.csv`:
```
conda activate venv
python correct_label.py --picks_file=picks.csv --num_source=96 --num_receiver=96 --p_pick_min=0 --p_pick_max=160 --s_pick_min=0 --s_pick_max=600 --epsP=1.75 --epsS=1.65 --CP=200 --CS=800 --plot_figure
```
Notes:
- `--num_source` and `--num_receiver` are the number of source and number of receiver in `picks.csv`. 
- `--p_pick_min`, `--p_pick_max`, `s_pick_min`, `--s_pick_max` allow to help to dectect anomalies by eliminating points that exceed upper and lower bounds.
- `--epsP` and `--epsS` control the number of samples that should be classified as outliers for P and S while using Huber regression in order to detect anomalies. The smaller the epsilon, the more robust it is to outliers.
- `--CP` and `--CS` are the regularization parameters of P and S while using SVR.