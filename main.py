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

# Read the epochs file: Lee el archivo de epochs usando mne
epochs_fname = config.pop('fname') 
epo = mne.read_epochs(epochs_fname)  

# Configuration depending on what we want
epo.pick_types(meg=True, eeg=False)

#Compute the evoked responses for two conditions: faces and scrambled
evoked_face = epo['face'].average()
evoked_scrambled = epo['scrambled'].average()

# Compute noise covariance matrix: Calcula la matriz de covarianza de ruido
noise_cov = mne.compute_covariance(epo, tmax=0.,
                                   method=['shrunk', 'empirical'],
                                   rank='info')
print(noise_cov['method'])

# == CONFIG PARAMETERS ==
fname_raw    = config['mne']
subjects_dir = config['output'] 
fname_trans  = config ['cov']
include_meg  = config['include_meg']
subject = 'output'

# == SOURCE SPACE ==
#Assume that coregistration is done
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir,add_dist=False)

# Compute BEM Model
conductivity = (0.3,)  # for single layer (MEG)
# conductivity = (0.3, 0.006, 0.3)  # for three layers (EEG)
model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# Compute Forward Model
fwd = mne.make_forward_solution(fname_raw, 
            trans=fname_trans,
            src=src, 
            bem=bem,
            meg=include_meg,  # include MEG channels
            eeg=False,  # exclude EEG channels
            mindist=5.0,  # ignore sources <= 5mm from inner skull
            n_jobs=1)  # number of jobs to run in parallel

#Fixed forward operator
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True)

#Compute the contrast between the two conditions
evoked_contrast = mne.combine_evoked([evoked_face, evoked_scrambled], [0.5, -0.5])
evoked_contrast.crop(-0.05, 0.25)
info = evoked_contrast.info
inverse_operator = make_inverse_operator (info, fwd_fixed, noise_cov, loose=0.2, depth=0.8)

#Applying inverse operator to our evoked contrast
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2  # regularization
stc = apply_inverse (evoked_contrast, inverse_operator, lambda2,  method=method, pick_ori=None)
print(stc)

# == SAVE RESULTS ==

# == SAVE SOURCE ESTIMATE ==
stc_fname = os.path.join('out_dir', 'stc')
stc.save(stc_fname)

# == SAVE SOURCE ESTIMATE FIGURE ==
subjects_dir = config.get('subjects_dir', os.path.join(__location__, 'freesurfer'))
fig_stc = stc.plot(hemi='both', subjects_dir=subjects_dir, subject='sub-01',
                   show_traces=False, views='lat', initial_time=0.1)

# Save the STC figure as an image
fig_stc_fname = os.path.join('out__figs', 'stc_plot.png')
fig_stc.savefig(fig_stc_fname)

# Create and save report
report = mne.Report(title='Inverse Operator Report')
report.add_figs_to_section(fig_stc, 'Source Estimate', section='STC')
report_path = os.path.join('out__dir_report', 'report.html')
report.save(report_path, overwrite=True)