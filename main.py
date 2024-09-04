# set up environment
import os
import json
import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mne.minimum_norm import (make_inverse_operator, apply_inverse)

# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Populate mne_config.py file with brainlife config.json
with open(__location__+'/config.json') as config_json:
    config = json.load(config_json)
    

# == config parameters fw ==
fname_fwd    = config['forward']
fname_noisecov  = config ['cov']
subject = 'output'

# == config parameters noisecov ==
fname = config ['epo']
epochs = mne.read_epochs(fname) 

# Configuration depending on what we want
epochs.pick_types(meg=True, eeg=False)

#Fixed forward operator
fwd_fixed = mne.convert_forward_solution(fname_fwd, surf_ori=True)

#Compute the evoked responses for two conditions: faces and scrambled
evoked = epochs.average()
info = evoked.info
inverse_operator = make_inverse_operator (info, fwd_fixed, fname_noisecov, loose=0.2, depth=0.8)

#Applying inverse operator to our evoked contrast
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2  # regularization
stc = apply_inverse (evoked, inverse_operator, lambda2,  method=method, pick_ori=None)
print(stc)


# == SAVE RESULTS ==

# == SAVE SOURCE ESTIMATE ==
stc_fname = os.path.join('out_dir', 'inv')
stc.save(stc_fname)

# == SAVE SOURCE ESTIMATE FIGURE ==
#fig_stc = stc.plot(hemi='both', subjects_dir=subjects_dir, subject='sub-01',
#                   show_traces=False, views='lat', initial_time=0.1)

# Save the STC figure as an image
#fig_stc_fname = os.path.join('out__figs', 'stc_plot.png')
#fig_stc.savefig(fig_stc_fname)

# Create and save report
report = mne.Report(title='Inverse Operator Report')
#report.add_figs_to_section(fig_stc, 'Source Estimate', section='STC')
report_path = os.path.join('out__dir_report', 'report.html')
report.save(report_path, overwrite=True)

