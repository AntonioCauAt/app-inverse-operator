<h1 align="center">Inverse operator</h1>

<p align="center">
  Builds an MNE-Python minimum-norm inverse operator from a forward solution, a noise covariance and epoched MEG data, and applies it (dSPM, SNR 3) to the averaged evoked response to output a source time course.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.830"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.830-blue.svg" alt="Run on brainlife.io"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Builds an MNE-Python minimum-norm inverse operator from a forward solution, a noise covariance and epoched MEG data, and applies it (dSPM, SNR 3) to the averaged evoked response to output a source time course.**

Averages the MEG channels of the epochs into an evoked response, builds an inverse operator with mne.minimum_norm.make_inverse_operator (loose 0.2, depth 0.8) from the forward solution and noise covariance, and applies dSPM with lambda2 = 1/9 to produce a source estimate (stc) and an HTML report. Use it for source localisation of evoked MEG responses. The loose, depth, fixed and verbose parameters are not read: the values are hard-coded in main.py, and EEG channels are dropped.

| | |
|---|---|
| **Inputs** | `neuro/meeg/mne/covariance` · `neuro/meeg/mne/epochs` · `neuro/meeg/mne/forward-inverse` |
| **Outputs** | `neuro/meeg/stc` · `report/html` · `generic/image/png` |
| **Container** | `brainlife/mne:1.2.1` |
| **Resources** | 8 CPUs |
| **Run it** | [doi.org/10.25663/brainlife.app.830](https://doi.org/10.25663/brainlife.app.830) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `cov1` | `neuro/meeg/mne/covariance` (matrix) | yes | Noise covariance matrix (MNE -cov.fif). |
| `evoked` | `neuro/meeg/mne/epochs` | yes | Epoched MEG data (-epo.fif); epochs are averaged into an evoked response. |
| `fif1` | `neuro/meeg/mne/forward-inverse` (fwd) | yes | MNE forward solution (-fwd.fif) for the same sensors and subject. |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `out_dir` | `neuro/meeg/stc` | Source time course (stc files, prefix inv) from dSPM. |
| `out_dir_report` | `report/html` | HTML report (report.html). |
| `out_figs` | `generic/image/png` | Figures of the source time course. |

<details>
<summary><b>Parameters</b> (4)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loose` | number | `0.2` | Value that weights the source variances of the dipole components that are parallel (tangential) to the cortical surface. Can be: float between 0 and 1 (inclusive) If 0, then the solution is computed with fixed orientation. If 1, it corresponds to free orientations. 'auto' (default) Uses 0.2 for surface source spaces (unless fixed is True) and 1.0 for other source spaces (volume or mixed). dict Mapping from the key for a given source space type (surface, volume, discrete) to the loose value. Useful mostly for mixed source spaces. |
| `depth` | number | `0.8` | How to weight (or normalize) the forward using a depth prior. If float (default 0.8), it acts as the depth weighting exponent (exp) to use None is equivalent to 0, meaning no depth weighting is performed. It can also be a dict containing keyword arguments to pass to mne.forward.compute_depth_prior()This is effectively ignored when method='eLORETA'. |
| `fixed` | boolean | `true` | Use fixed source orientations normal to the cortical mantle. If True, the loose parameter must be "auto" or 0. If 'auto', the loose value is used. |
| `verbose` | string | `None` | Control verbosity of the logging output. If None, use the default verbosity level. See the logging documentation and mne.verbose() for details. Should only be passed as a keyword argument. |

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.830](https://doi.org/10.25663/brainlife.app.830), choose `neuro/meeg/mne/covariance`, `neuro/meeg/mne/epochs`, `neuro/meeg/mne/forward-inverse` from your project, set the parameters and press **Execute**. The results are saved to your project as `neuro/meeg/stc`, `report/html`, `generic/image/png`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `brainlife/mne:1.2.1`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "cov": "/path/to/cov1/cov",
       "epo": "/path/to/evoked/epo",
       "forward": "/path/to/fif1/forward",
       "inverse": "/path/to/fif1/inverse",
       "loose": 0.2,
       "depth": 0.8,
       "fixed": true,
       "verbose": "None"
   }
   ```

3. Run `./main`.

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- Inverse operator. brainlife.io App. [doi:10.25663/brainlife.app.830](https://doi.org/10.25663/brainlife.app.830)
- Gramfort A. et al. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience 7:267 (2013). [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Guiomar Niso
- Antonio Caulín

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io, which runs this App, is publicly funded by Wellcome, the U.S. National Institutes of Health and the U.S. National Science Foundation; acknowledging that support helps keep it running.</sub>
