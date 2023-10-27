#### CMRversions
CMR model code for a free recall paradigm (as described in Zhang et al., 2022).
Zhang, Q., Griffiths, T. L., & Norman, K. A. (2022). Optimal policies for free recall. Psychological Review. Advance online publication. https://doi.org/10.1037/rev0000375

#### data
Folder containing the pilot and final datasets (includes raw files, clean files, text files used for analysis, and relevant code).

#### fitCMR
Folder containing code used to estimate CMR parameters to a pilot dataset. See the folder for more detail.

#### final_analysis.ipynb; more_analysis.ipynb
Pre-registered and exploratory analysis and model simulations reported in manuscript (further documentation in notebook; final_analysis has python kernal, more_analysis has R kernal for linear models).

#### analysis_helpers.ipynb
Helper functions for generating output in analysis notebooks.

#### functions_overrides.py; probCMR_overrides.py
These two files contain the code to simulate a post-cue recall session that override specific functions from CMRversions. In functions_overrides.py, this allows for (1) choosing which datafiles to be run and (2) sequencing the code (probCMR_overrides.py) to simulate a reminder and post-cue recall session after a free recall session. See the code for further documentation.



If you find the codes useful, please cite:
Cornell, C.A., Norman, K.A., Griffiths, T.L., & Zhang, Q. (submitted + preprint). Improving memory search through model-based cue selection. https://psyarxiv.com/atqs7