Correcting predicted picks in `picks.csv`:

`python correct_label.py --picks_file=picks.csv --num_source=96 --num_receiver=96 --p_pick_min=0 --p_pick_max=160 --s_pick_min=0 --s_pick_max=600 --std_coef=0.4 --plot_figure`

Notes:
- The parameters `--p_pick_min`, `--p_pick_max`, `s_pick_min`, `--s_pick_max` allow to help to dectect anomalies by eliminating points that exceed upper and lower bounds.
- The parameter `--std_coef` allows to determine an interpolation zone in which the interpolated points oscillate less.
