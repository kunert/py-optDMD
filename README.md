# pyoptDMD
A Python implementation of code for computing optimized DMD.
This code is converted from a MATLAB version by Travis Askham, available here: 
https://github.com/duqbo/optdmd

The code implements the optimized DMD as described in "Variable Projection Methods for an 
Optimized Dynamic Mode Decomposition" by Askham and Kutz.

It still needs to be demonstrated if the python version is a faithful reconstruction 
of the original optdmd code.

### Current status:

The code successfully solves for the dmd modes for noiseless data but encounters 
convergence errors for the noisy cases.