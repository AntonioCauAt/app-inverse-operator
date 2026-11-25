
# Inverse Operator 

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Builds an MNE-Python minimum-norm inverse operator from a forward solution, a noise covariance and epoched MEG data, and applies it (dSPM, SNR 3) to the averaged evoked response to output a source time course.**

Averages the MEG channels of the epochs into an evoked response, builds an inverse operator with mne.minimum_norm.make_inverse_operator (loose 0.2, depth 0.8) from the forward solution and noise covariance, and applies dSPM with lambda2 = 1/9 to produce a source estimate (stc) and an HTML report. Use it for source localisation of evoked MEG responses. The loose, depth, fixed and verbose parameters are not read: the values are hard-coded in main.py, and EEG channels are dropped.

| | |
|---|---|
| 📥 **Inputs** | `neuro/meeg/mne/covariance` · `neuro/meeg/mne/epochs` · `neuro/meeg/mne/forward-inverse` |
| 📤 **Outputs** | `neuro/meeg/stc` · `report/html` · `generic/image/png` |
| 🐳 **Container** | `brainlife/mne:1.2.1` |
| ⚙️ **Resources** | 8 CPUs |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.830](https://doi.org/10.25663/brainlife.app.830) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `cov1` | `neuro/meeg/mne/covariance` (matrix) | yes | Noise covariance matrix (MNE -cov.fif). |
| `evoked` | `neuro/meeg/mne/epochs` | yes | Epoched MEG data (-epo.fif); epochs are averaged into an evoked response. |
| `fif1` | `neuro/meeg/mne/forward-inverse` (fwd) | yes | MNE forward solution (-fwd.fif) for the same sensors and subject. |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `out_dir` | `neuro/meeg/stc` | Source time course (stc files, prefix inv) from dSPM. |
| `out_dir_report` | `report/html` | HTML report (report.html). |
| `out_figs` | `generic/image/png` | Figures of the source time course. |

<details>
<summary><b>⚙️ Parameters</b> (4)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loose` | number | `0.2` | Value that weights the source variances of the dipole components that are parallel (tangential) to the cortical surface. Can be: float between 0 and 1 (inclusive) If 0, then the solution is computed with fixed orientation. If 1, it corresponds to free orientations. 'auto' (default) Uses 0.2 for surface source spaces (unless fixed is True) and 1.0 for other source spaces (volume or mixed). dict Mapping from the key for a given source space type (surface, volume, discrete) to the loose value. Useful mostly for mixed source spaces. |
| `depth` | number | `0.8` | How to weight (or normalize) the forward using a depth prior. If float (default 0.8), it acts as the depth weighting exponent (exp) to use None is equivalent to 0, meaning no depth weighting is performed. It can also be a dict containing keyword arguments to pass to mne.forward.compute_depth_prior()This is effectively ignored when method='eLORETA'. |
| `fixed` | boolean | `true` | Use fixed source orientations normal to the cortical mantle. If True, the loose parameter must be "auto" or 0. If 'auto', the loose value is used. |
| `verbose` | string | `None` | Control verbosity of the logging output. If None, use the default verbosity level. See the logging documentation and mne.verbose() for details. Should only be passed as a keyword argument. |

</details>

📚 **References used by this App:**
- Avesani P. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)
- Gramfort A. et al. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience 7:267 (2013). [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)

<!-- brainlife-app:end -->

[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.826-blue.svg)](https://doi.org/10.25663/brainlife.app.826)

Brainlife App to compute the Inverse Operator using the `mne.minimum_norm.make_inverse_operator` function, which estimates the brain sources from MEG data by applying an inverse solution based on the forward model, noise covariance, and evoked responses and it is apply obtaing a source estimate



1) Input file is:
    * `cov` covariance data file
    * `epochs`  evoked data file
    * `fif` fwd data file

2) Input parameters are:
* `loose` Value that weights the source variance.
* `depth` Weight of the forward.
* `fixed` Source Orientations.
* `Rank` Control the rank computation.
* `Use_cps` Use of cortical paths.
* `Verbose` Control verboisity of the logging output

3) Ouput files are:
    * Source Time Course
    * a plot of the STC
    * a html report
   

## Authors
- Guiomar Niso (guiomar.niso@gmail.com)
- Antonio Caulín (antoniocaulinatienzar@gmail.com)

## Citations
We kindly ask that you cite the following articles when publishing papers and code using this code. 

*- brainlife.io Publishing and Apps:*

Avesani, P., McPherson, B., Hayashi, S. et al. **The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services**. Sci Data 6, 69 (2019). https://doi.org/10.1038/s41597-019-0073-y

*- MNE-Python package:* 

Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Goj R, Jas M, Brooks T, Parkkonen L, and Hämäläinen MS.  **MEG and EEG data analysis with MNE-Python**. Frontiers in Neuroscience, 7(267):1–13, 2013. https://doi.org/10.3389/fnins.2013.00267

## Funding Acknowledgement
brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your publications and code reusing this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://grantome.com/grant/NIH/R01-EB030896-01)


#### MIT Copyright (c) 2021 brainlife.io The University of Texas at Austin and Indiana University